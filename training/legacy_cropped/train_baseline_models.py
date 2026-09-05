"""
train_baseline_models.py
=========================
Trains 4 ADDITIONAL baseline models that are architecturally different
from the U-Net family (Models A-E in train_all_models.py). These exist
to prove that Model E's design choices (residual blocks + attention
gates + ASPP + composite loss) beat not just weaker U-Nets, but also
other well-known inpainting/reconstruction architecture families.

  Model F — Plain Autoencoder     5ch input | no skip connections at all
                                              | plain MSE
                                              -> isolates the value of skip
                                                 connections (vs Models A-E)

  Model G — Context-Encoder GAN   5ch input | encoder-decoder generator
                                              + PatchGAN discriminator
                                              | L2 (region-weighted) + adversarial
                                              -> classic adversarial inpainting
                                                 baseline (Pathak et al., 2016 style)

  Model H — Partial-Conv U-Net    5ch input | PartialConv2d layers, mask-aware
                                              convolutions, skip connections
                                              | hole L1 + valid L1 + TV loss
                                              -> occlusion-specific baseline
                                                 (Liu et al., 2018 style)

  Model I — ViT Inpainter         5ch input | patch embedding + transformer
                                              encoder + linear un-patchify
                                              | region-weighted MSE + L1
                                              -> global-context transformer
                                                 baseline (no convolutional
                                                 inductive bias at all)

Each model is saved in its own folder, using the SAME train/test split
JSON as train_all_models.py so results are directly comparable.

Usage:
    python train_baseline_models.py
"""

import os, json, csv, time, math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

# ═══════════════════════════════════════════════════════════════
# CONFIG  (identical to train_all_models.py for a fair comparison)
# ═══════════════════════════════════════════════════════════════

SPLIT_PATH   = os.path.join("traintestsplit", "data_split.json")
TARGET_SIZE  = (256, 256)
BATCH_SIZE   = 8
EPOCHS       = 60
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 12
NUM_WORKERS  = 0
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Region loss weights (used by Models G, I; H uses PConv's own hole/valid split)
W_HIDDEN  = 1.0
W_VISIBLE = 0.4
W_BG      = 0.1

# GAN-specific
LAMBDA_L2  = 100.0   # reconstruction weight vs adversarial weight in Model G
LAMBDA_ADV = 1.0

print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════
# SPLIT LOADER  (same format as train_all_models.py)
# ═══════════════════════════════════════════════════════════════

def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split not found: {path}\nRun create_split.py first.")
    with open(path) as f:
        split = json.load(f)
    train = split["train"]
    test  = split.get("test", split.get("val", []))
    print(f"Split loaded — train: {len(train)}  test: {len(test)}")
    return train, test


# ═══════════════════════════════════════════════════════════════
# DATASET  (identical to train_all_models.py — 5ch input for all
# baselines here, so every model sees RGB + occlusion mask + bbox mask)
# ═══════════════════════════════════════════════════════════════

class SignDataset(Dataset):
    def __init__(self, samples, target_size=(256, 256), in_ch=5, augment=False):
        self.samples     = samples
        self.target_size = target_size
        self.in_ch       = in_ch
        self.augment     = augment

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def _load_mask(self, path):
        if not path or not os.path.exists(path):
            H, W = self.target_size[1], self.target_size[0]
            return np.zeros((H, W), dtype=np.float32)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            H, W = self.target_size[1], self.target_size[0]
            return np.zeros((H, W), dtype=np.float32)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)

    def _augment(self, occ, clean, occ_mask):
        if np.random.rand() > 0.5:
            occ      = occ[:, ::-1, :].copy()
            clean    = clean[:, ::-1, :].copy()
            occ_mask = occ_mask[:, ::-1].copy()
        if np.random.rand() > 0.8:
            occ      = occ[::-1, :, :].copy()
            clean    = clean[::-1, :, :].copy()
            occ_mask = occ_mask[::-1, :].copy()
        f = np.random.uniform(0.85, 1.15)
        occ   = np.clip(occ * f, 0, 1)
        clean = np.clip(clean * f, 0, 1)
        return occ, clean, occ_mask

    def __getitem__(self, idx):
        s        = self.samples[idx]
        occ_img  = np.clip(self._load_rgb(s['occ_path']), 0, 1)
        clean    = np.clip(self._load_rgb(s['clean_path']), 0, 1)
        occ_mask = np.clip(self._load_mask(s.get('mask_path', '')), 0, 1)

        if self.augment:
            occ_img, clean, occ_mask = self._augment(occ_img, clean, occ_mask)

        bbox_mask = np.clip(
            cv2.dilate(occ_mask, np.ones((15, 15), np.uint8), iterations=1), 0, 1)
        x = np.concatenate([occ_img, occ_mask[..., None], bbox_mask[..., None]], axis=-1)

        return (
            torch.from_numpy(x.transpose(2, 0, 1)).float(),
            torch.from_numpy(clean.transpose(2, 0, 1)).float(),
            torch.from_numpy(occ_mask).unsqueeze(0).float(),
        )


def collate_fn(batch):
    xs, ys, ms = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


# ═══════════════════════════════════════════════════════════════
# MODEL F — PLAIN AUTOENCODER  (no skip connections at all)
# ═══════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class PlainAutoencoder(nn.Module):
    """
    Model F — Plain Autoencoder.
    Same depth/channel progression as the U-Net family (64-128-256-512-1024)
    but the decoder receives NOTHING from the encoder except the bottleneck.
    This isolates exactly how much U-Net's skip connections are worth.
    """
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        self.e1 = ConvBlock(in_ch, 64);   self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(64, 128);      self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(128, 256);     self.p3 = nn.MaxPool2d(2)
        self.e4 = ConvBlock(256, 512);     self.p4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)

        self.d4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 2, stride=2), ConvBlock(512, 512))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2),  ConvBlock(256, 256))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2),  ConvBlock(128, 128))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),   ConvBlock(64, 64))
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x = self.p1(self.e1(x))
        x = self.p2(self.e2(x))
        x = self.p3(self.e3(x))
        x = self.p4(self.e4(x))
        x = self.bottleneck(x)
        x = self.d4(x)
        x = self.d3(x)
        x = self.d2(x)
        x = self.d1(x)
        return torch.sigmoid(self.out(x))


def loss_F(pred, gt, occ_mask):
    """Model F — plain MSE, no region weighting (deliberately the simplest baseline)."""
    return F.mse_loss(pred, gt)


# ═══════════════════════════════════════════════════════════════
# MODEL G — CONTEXT-ENCODER GAN
# ═══════════════════════════════════════════════════════════════

class CEGenerator(nn.Module):
    """Encoder-decoder generator with a bottleneck, in the spirit of
    Pathak et al.'s Context Encoder (channel-reduced conv bottleneck
    instead of the original's fully-connected one, for memory reasons
    at 256x256 resolution)."""
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        def down(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.LeakyReLU(0.2, inplace=True))
        def up(ci, co):
            return nn.Sequential(
                nn.ConvTranspose2d(ci, co, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True))

        self.enc = nn.Sequential(
            down(in_ch, 64), down(64, 128), down(128, 256),
            down(256, 512), down(512, 512),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(
            up(512, 512), up(512, 256), up(256, 128), up(128, 64),
            nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        z = self.bottleneck(z)
        out = self.dec(z)
        return torch.sigmoid(out)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""
    def __init__(self, in_ch=3):
        super().__init__()
        def block(ci, co, norm=True, stride=2):
            layers = [nn.Conv2d(ci, co, 4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def region_weights(occ_mask):
    """Shared with Model I — hidden region weighted highest, then visible, then bg."""
    occ = occ_mask.float()
    return W_HIDDEN * occ + W_VISIBLE * 0 + W_BG * (1 - occ) + W_VISIBLE * (occ * 0)
    # (kept simple/explicit below instead — see loss_G / loss_I)


def region_weight_map(occ_mask):
    occ = occ_mask.float()
    return W_HIDDEN * occ + W_BG * (1.0 - occ)


def loss_G_recon(pred, gt, occ_mask):
    """Reconstruction half of Model G's loss: region-weighted L2."""
    w = region_weight_map(occ_mask).expand_as(pred)
    return (w * (pred - gt) ** 2).mean()


# ═══════════════════════════════════════════════════════════════
# MODEL H — PARTIAL-CONVOLUTION U-NET
# ═══════════════════════════════════════════════════════════════

class PartialConv2d(nn.Conv2d):
    """
    Mask-aware convolution (Liu et al., 2018).
    Output at a location is renormalized by how many valid (unmasked)
    input pixels contributed to it, and the mask is updated so a
    location becomes 'valid' as soon as the receptive field touches
    ANY valid pixel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            'weight_mask_updater',
            torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.slide_win_size = self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x, mask_in):
        with torch.no_grad():
            update_mask = F.conv2d(
                mask_in, self.weight_mask_updater.to(x.dtype),
                bias=None, stride=self.stride, padding=self.padding)
            mask_ratio = self.slide_win_size / (update_mask + 1e-8)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask

        raw_out = super().forward(x * mask_in)

        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            out = (raw_out - bias_view) * mask_ratio + bias_view
            out = out * update_mask
        else:
            out = raw_out * mask_ratio

        return out, update_mask


class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, norm=True, act='relu'):
        super().__init__()
        pad = kernel // 2
        self.pconv = PartialConv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=not norm)
        self.bn = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.act(self.bn(x))
        return x, mask


class PConvUNet(nn.Module):
    """
    Model H — Partial-Convolution U-Net.
    Encoder/decoder with PartialConv2d throughout; both the features
    AND the mask are propagated and concatenated at each skip connection.
    """
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        self.e1 = PConvBlock(in_ch, 64, kernel=7, stride=2)
        self.e2 = PConvBlock(64, 128, kernel=5, stride=2)
        self.e3 = PConvBlock(128, 256, kernel=3, stride=2)
        self.e4 = PConvBlock(256, 512, kernel=3, stride=2)
        self.e5 = PConvBlock(512, 512, kernel=3, stride=2)

        self.d5 = PConvBlock(512 + 512, 512, act='leaky')
        self.d4 = PConvBlock(512 + 256, 256, act='leaky')
        self.d3 = PConvBlock(256 + 128, 128, act='leaky')
        self.d2 = PConvBlock(128 + 64, 64, act='leaky')
        self.d1 = PConvBlock(64 + in_ch, 32, act='leaky')
        self.out_conv = nn.Conv2d(32, out_ch, 1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def forward(self, x, mask):
        # mask: 1 = valid/visible, 0 = hole. Single channel throughout —
        # PartialConv2d broadcasts it against however many feature channels
        # the layer has, so it never needs to be expanded to match x's channels.
        m0 = mask

        x1, m1 = self.e1(x, m0)
        x2, m2 = self.e2(x1, m1)
        x3, m3 = self.e3(x2, m2)
        x4, m4 = self.e4(x3, m3)
        x5, m5 = self.e5(x4, m4)

        u5 = self._up(x5, x4.shape[-2:]); um5 = self._up(m5, m4.shape[-2:])
        d5, dm5 = self.d5(torch.cat([u5, x4], 1), torch.max(um5, m4))

        u4 = self._up(d5, x3.shape[-2:]); um4 = self._up(dm5, m3.shape[-2:])
        d4, dm4 = self.d4(torch.cat([u4, x3], 1), torch.max(um4, m3))

        u3 = self._up(d4, x2.shape[-2:]); um3 = self._up(dm4, m2.shape[-2:])
        d3, dm3 = self.d3(torch.cat([u3, x2], 1), torch.max(um3, m2))

        u2 = self._up(d3, x1.shape[-2:]); um2 = self._up(dm3, m1.shape[-2:])
        d2, dm2 = self.d2(torch.cat([u2, x1], 1), torch.max(um2, m1))

        u1 = self._up(d2, x.shape[-2:]); um1 = self._up(dm2, m0.shape[-2:])
        d1, _ = self.d1(torch.cat([u1, x], 1), torch.max(um1, m0))

        return torch.sigmoid(self.out_conv(d1))


def total_variation_loss(pred, hole_mask):
    """Smoothness penalty at hole boundaries (simplified vs. the paper's
    dilated-hole-region TV — here applied over the full hole region)."""
    m = hole_mask
    tv_h = ((pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs() * m[:, :, 1:, :]).mean()
    tv_w = ((pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs() * m[:, :, :, :-1]).mean()
    return tv_h + tv_w


def loss_H(pred, gt, occ_mask):
    """
    Model H — PConv-style loss.
    hole = occluded region (occ_mask == 1), valid = everything else.
    NOTE: the original paper also adds VGG perceptual + style losses;
    those are omitted here since they require a pretrained VGG that
    may not be available offline. hole/valid L1 + TV already captures
    most of PConv's practical behaviour.
    """
    hole  = occ_mask.expand_as(pred)
    valid = 1.0 - hole
    n_hole  = hole.sum().clamp(min=1)
    n_valid = valid.sum().clamp(min=1)

    l_hole  = ((pred - gt).abs() * hole).sum() / n_hole
    l_valid = ((pred - gt).abs() * valid).sum() / n_valid
    l_tv    = total_variation_loss(pred, hole)

    return 6.0 * l_hole + 1.0 * l_valid + 0.1 * l_tv


# ═══════════════════════════════════════════════════════════════
# MODEL I — VIT INPAINTER  (pure transformer, no convolutional
# inductive bias — the architectural opposite extreme from Model E)
# ═══════════════════════════════════════════════════════════════

class ViTInpainter(nn.Module):
    def __init__(self, in_ch=5, out_ch=3, img_size=256, patch=16,
                 embed_dim=384, depth=6, n_heads=6, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        self.patch      = patch
        self.grid       = img_size // patch
        self.n_patches  = self.grid * self.grid
        self.out_ch     = out_ch

        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        # un-patchify: each token -> patch*patch*out_ch pixel values
        self.head = nn.Linear(embed_dim, patch * patch * out_ch)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)                       # (B, E, grid, grid)
        tokens = tokens.flatten(2).transpose(1, 2)          # (B, N, E)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        pix = self.head(tokens)                             # (B, N, patch*patch*out_ch)
        pix = pix.view(B, self.grid, self.grid, self.patch, self.patch, self.out_ch)
        pix = pix.permute(0, 5, 1, 3, 2, 4).contiguous()     # (B, out_ch, grid, patch, grid, patch)
        img = pix.view(B, self.out_ch, self.grid * self.patch, self.grid * self.patch)
        return torch.sigmoid(img)


def loss_I(pred, gt, occ_mask):
    """Model I — region-weighted MSE + L1 on the hidden region (same
    recipe as Model D, so any gap vs Model E is attributable to the
    architecture, not the loss)."""
    w = region_weight_map(occ_mask).expand_as(pred)
    mse = (w * (pred - gt) ** 2).mean()
    hole = occ_mask.expand_as(pred)
    n = hole.sum().clamp(min=1)
    l1 = ((pred - gt).abs() * hole).sum() / n
    return 0.7 * mse + 0.3 * l1


# ═══════════════════════════════════════════════════════════════
# SAVE HELPERS
# ═══════════════════════════════════════════════════════════════

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_csv(history, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        cols = list(history.keys())
        w.writerow(['epoch'] + cols)
        n = len(history[cols[0]])
        for i in range(n):
            w.writerow([i + 1] + [f"{history[c][i]:.6f}" for c in cols])

def save_curve(history, path, model_name):
    if not _HAVE_PLT:
        return
    plt.figure(figsize=(8, 4))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.title(f'{model_name} — loss curve')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


# ═══════════════════════════════════════════════════════════════
# TRAIN — PLAIN / PCONV / VIT (single-network, single-loss models)
# ═══════════════════════════════════════════════════════════════

def train_plain_model(cfg, train_loader, val_loader, save_dir):
    arch     = cfg['arch']
    loss_fn  = cfg['loss_fn']

    if arch == 'plain_ae':
        model = PlainAutoencoder(in_ch=cfg['in_ch'], out_ch=3).to(DEVICE)
    elif arch == 'pconv':
        model = PConvUNet(in_ch=cfg['in_ch'], out_ch=3).to(DEVICE)
    elif arch == 'vit':
        model = ViTInpainter(in_ch=cfg['in_ch'], out_ch=3,
                              img_size=TARGET_SIZE[0]).to(DEVICE)
    else:
        raise ValueError(f"Unknown arch {arch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train': [], 'val': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, skipped = 0.0, 0

        for x, y, occ_mask in train_loader:
            x, y, occ_mask = x.to(DEVICE), y.to(DEVICE), occ_mask.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if arch == 'pconv':
                # visible mask = 1 - occlusion mask (PConv convention: 1=valid)
                valid_mask = 1.0 - occ_mask
                pred = model(x, valid_mask)   # full 5ch input, same as other baselines
            else:
                pred = model(x)

            loss = loss_fn(pred, y, occ_mask)
            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        n_valid = max(1, len(train_loader) - skipped)
        train_loss /= n_valid

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, occ_mask in val_loader:
                x, y, occ_mask = x.to(DEVICE), y.to(DEVICE), occ_mask.to(DEVICE)
                if arch == 'pconv':
                    valid_mask = 1.0 - occ_mask
                    pred = model(x, valid_mask)
                else:
                    pred = model(x)
                loss = loss_fn(pred, y, occ_mask)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history['train'].append(float(train_loss))
        history['val'].append(float(val_loss))
        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:03d}/{EPOCHS}  train={train_loss:.5f}  "
              f"val={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, best_epoch, patience_counter = val_loss, epoch + 1, 0
            torch.save({'model_name': cfg['name'], 'arch': arch, 'in_ch': cfg['in_ch'],
                        'model_state_dict': model.state_dict(), 'best_val': best_val,
                        'history': history}, os.path.join(save_dir, cfg['ckpt']))
            print(f"    ✓ Best model saved  (val={best_val:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"  Early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    torch.save({'model_name': cfg['name'], 'arch': arch, 'in_ch': cfg['in_ch'],
                'model_state_dict': model.state_dict(), 'history': history},
               os.path.join(save_dir, cfg['ckpt'].replace('best', 'final')))
    return history, best_val, best_epoch, elapsed, n_params


# ═══════════════════════════════════════════════════════════════
# TRAIN — MODEL G (GAN: alternating generator / discriminator updates)
# ═══════════════════════════════════════════════════════════════

def train_gan_model(cfg, train_loader, val_loader, save_dir):
    G = CEGenerator(in_ch=cfg['in_ch'], out_ch=3).to(DEVICE)
    D = PatchDiscriminator(in_ch=3).to(DEVICE)

    n_params = sum(p.numel() for p in G.parameters() if p.requires_grad) + \
               sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"  Parameters (G+D) : {n_params:,}")

    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    bce = nn.BCEWithLogitsLoss()

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train_g': [], 'train_d': [], 'val': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        G.train(); D.train()
        g_loss_sum, d_loss_sum, skipped = 0.0, 0.0, 0

        for x, y, occ_mask in train_loader:
            x, y, occ_mask = x.to(DEVICE), y.to(DEVICE), occ_mask.to(DEVICE)
            b = x.size(0)
            real_label = torch.ones(D(y).shape, device=DEVICE)
            fake_label = torch.zeros_like(real_label)

            # ---- Discriminator step ----
            with torch.no_grad():
                fake = G(x)
            opt_d.zero_grad(set_to_none=True)
            d_real = D(y)
            d_fake = D(fake.detach())
            d_loss = 0.5 * (bce(d_real, real_label) + bce(d_fake, fake_label))
            d_loss.backward()
            opt_d.step()

            # ---- Generator step ----
            opt_g.zero_grad(set_to_none=True)
            fake = G(x)
            d_fake_for_g = D(fake)
            adv_loss = bce(d_fake_for_g, real_label)
            recon_loss = loss_G_recon(fake, y, occ_mask)
            g_loss = LAMBDA_L2 * recon_loss + LAMBDA_ADV * adv_loss

            if not torch.isfinite(g_loss):
                skipped += 1
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

        n_valid = max(1, len(train_loader) - skipped)
        train_g = g_loss_sum / n_valid
        train_d = d_loss_sum / n_valid

        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, occ_mask in val_loader:
                x, y, occ_mask = x.to(DEVICE), y.to(DEVICE), occ_mask.to(DEVICE)
                pred = G(x)
                loss = loss_G_recon(pred, y, occ_mask)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history['train_g'].append(float(train_g))
        history['train_d'].append(float(train_d))
        history['val'].append(float(val_loss))
        sched_g.step(val_loss)

        print(f"  Epoch {epoch+1:03d}/{EPOCHS}  G={train_g:.5f}  D={train_d:.5f}  "
              f"val(recon)={val_loss:.5f}  lr={opt_g.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, best_epoch, patience_counter = val_loss, epoch + 1, 0
            torch.save({'model_name': cfg['name'], 'arch': cfg['arch'], 'in_ch': cfg['in_ch'],
                        'generator_state_dict': G.state_dict(),
                        'discriminator_state_dict': D.state_dict(),
                        'best_val': best_val, 'history': history},
                       os.path.join(save_dir, cfg['ckpt']))
            print(f"    ✓ Best model saved  (val={best_val:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"  Early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    torch.save({'model_name': cfg['name'], 'arch': cfg['arch'], 'in_ch': cfg['in_ch'],
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(), 'history': history},
               os.path.join(save_dir, cfg['ckpt'].replace('best', 'final')))
    return history, best_val, best_epoch, elapsed, n_params


# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════

MODELS = [
    {
        "name"    : "Model_F_Plain_Autoencoder",
        "in_ch"   : 5,
        "arch"    : "plain_ae",
        "loss_fn" : loss_F,
        "save_dir": "model_F_plain_autoencoder",
        "ckpt"    : "best_model_F.pth",
    },
    {
        "name"    : "Model_G_ContextEncoder_GAN",
        "in_ch"   : 5,
        "arch"    : "gan",
        "loss_fn" : None,   # handled inside train_gan_model
        "save_dir": "model_G_context_encoder_gan",
        "ckpt"    : "best_model_G.pth",
    },
    {
        "name"    : "Model_H_PartialConv_UNet",
        "in_ch"   : 5,
        "arch"    : "pconv",
        "loss_fn" : loss_H,
        "save_dir": "model_H_partial_conv_unet",
        "ckpt"    : "best_model_H.pth",
    },
    {
        "name"    : "Model_I_ViT_Inpainter",
        "in_ch"   : 5,
        "arch"    : "vit",
        "loss_fn" : loss_I,
        "save_dir": "model_I_vit_inpainter",
        "ckpt"    : "best_model_I.pth",
    },
]


# ═══════════════════════════════════════════════════════════════
# TRAIN ONE MODEL (dispatcher)
# ═══════════════════════════════════════════════════════════════

def train_model(cfg, train_samples, val_samples):
    name, save_dir = cfg['name'], cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Training : {name}")
    print(f"  Arch     : {cfg['arch'].upper()}")
    print(f"  Input ch : {cfg['in_ch']}")
    print(f"  Save dir : {save_dir}/")
    print(f"{'='*64}")

    train_ds = SignDataset(train_samples, TARGET_SIZE, in_ch=cfg['in_ch'], augment=True)
    val_ds   = SignDataset(val_samples, TARGET_SIZE, in_ch=cfg['in_ch'], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    if cfg['arch'] == 'gan':
        history, best_val, best_epoch, elapsed, n_params = train_gan_model(cfg, train_loader, val_loader, save_dir)
    else:
        history, best_val, best_epoch, elapsed, n_params = train_plain_model(cfg, train_loader, val_loader, save_dir)

    save_json(os.path.join(save_dir, 'history.json'), history)
    save_csv(history, os.path.join(save_dir, 'epoch_log.csv'))
    save_curve(history, os.path.join(save_dir, 'loss_curve.png'), name)
    save_json(os.path.join(save_dir, 'training_summary.json'), {
        'model_name': name, 'arch': cfg['arch'], 'in_ch': cfg['in_ch'],
        'best_val_loss': best_val, 'best_epoch': best_epoch,
        'epochs_completed': len(history.get('val', [])),
        'duration_seconds': elapsed, 'parameters': n_params,
    })

    print(f"\n  Finished {name}")
    print(f"  Best val : {best_val:.5f} at epoch {best_epoch}")
    print(f"  Time     : {elapsed/60:.1f} min")
    print(f"  Saved to : {save_dir}/")
    return best_val, best_epoch


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading split...")
    train_samples, val_samples = load_split(SPLIT_PATH)

    results = []
    for cfg in MODELS:
        best_val, best_epoch = train_model(cfg, train_samples, val_samples)
        results.append({'model': cfg['name'], 'best_val': best_val,
                         'best_epoch': best_epoch, 'save_dir': cfg['save_dir']})

    print(f"\n{'='*64}")
    print("  ALL BASELINE MODELS TRAINING COMPLETE")
    print(f"{'='*64}")
    print(f"  {'Model':<30} {'Best Val':>10} {'Epoch':>7}")
    print("  " + "-" * 52)
    for r in results:
        print(f"  {r['model']:<30} {r['best_val']:>10.5f} {r['best_epoch']:>7}")

    save_json("training_baseline_results.json", results)
    print("\n  Summary saved -> training_baseline_results.json")
    print("\nNext step: run evaluate_all_models.py on models F-I the same way")
    print("as models A-E, then compare all 9 models' PSNR/SSIM/MS-SSIM/LPIPS")
    print("tables — Model E should top every architecture family.")


if __name__ == "__main__":
    main()
