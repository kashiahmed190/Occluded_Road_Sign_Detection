import os
import json
import csv
import time
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================================================
# MODEL H — Partial-Convolution U-Net
#   Input  : 5 channels (RGB + occ_mask + bbox_mask)
#   Arch   : PartialConv2d layers (mask-aware convolutions),
#            U-Net-style skip connections
#   Loss   : hole L1 + valid L1 + total-variation smoothness
#   Purpose: occlusion-specific baseline (Liu et al., 2018 style)
#            -- different architecture family from the plain U-Nets
# =========================================================

OCCLUDED_DIR    = "occ"
MASK_DIR        = "occ_masks"
CLEAN_DIR       = "d"
ANNOT_DIR       = "e"

BASE_SPLIT_PATH = os.path.join("traintestsplit", "data_split.json")
SAVE_DIR        = "model_H_partial_conv_unet"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_SIZE  = (256, 256)
BATCH_SIZE   = 8
EPOCHS       = 60
LR           = 1e-4
NUM_WORKERS  = 2
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 12
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH   = os.path.join(SAVE_DIR, "best_model_H.pth")
FINAL_MODEL_PATH  = os.path.join(SAVE_DIR, "final_model_H.pth")
LOSS_PLOT_PATH    = os.path.join(SAVE_DIR, "loss_curve.png")
HISTORY_JSON_PATH = os.path.join(SAVE_DIR, "history.json")
HISTORY_CSV_PATH  = os.path.join(SAVE_DIR, "epoch_log.csv")
CONFIG_JSON_PATH  = os.path.join(SAVE_DIR, "training_config.json")
SUMMARY_JSON_PATH = os.path.join(SAVE_DIR, "training_summary.json")
SPLIT_COPY_PATH   = os.path.join(SAVE_DIR, "split_used.json")

print("Device:", DEVICE)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def load_existing_split(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        split = json.load(f)
    train_samples = split["train"]
    val_samples   = split.get("test", split.get("val", []))
    print(f"Split loaded: train={len(train_samples)}  val={len(val_samples)}\n")
    return train_samples, val_samples, split


def parse_annotation(xml_path, original_hw, target_hw):
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        H0, W0 = original_hw
        Ht, Wt = target_hw
        sx = Wt / float(W0)
        sy = Ht / float(H0)
        bboxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = int(float(bb.find("xmin").text) * sx)
            ymin = int(float(bb.find("ymin").text) * sy)
            xmax = int(float(bb.find("xmax").text) * sx)
            ymax = int(float(bb.find("ymax").text) * sy)
            xmin = max(0, min(Wt-1, xmin))
            xmax = max(0, min(Wt,   xmax))
            ymin = max(0, min(Ht-1, ymin))
            ymax = max(0, min(Ht,   ymax))
            if xmax > xmin and ymax > ymin:
                bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")
        return []

def make_bbox_mask(h, w, bboxes):
    mask = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 1.0
    return mask


class PConvDataset(Dataset):
    def __init__(self, samples, annot_dir, mask_dir,
                 target_size=(256, 256), augment=False):
        self.samples     = samples
        self.annot_dir   = annot_dir
        self.mask_dir    = mask_dir
        self.target_size = target_size
        self.augment     = augment

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read: {path}")
        H0, W0 = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0, (H0, W0)

    def _load_occ_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            Ht, Wt = self.target_size[1], self.target_size[0]
            return np.zeros((Ht, Wt), dtype=np.float32)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)

    def _augment(self, occ, clean, occ_mask, bbox_mask):
        if np.random.rand() > 0.5:
            occ       = occ      [:, ::-1, :].copy()
            clean     = clean    [:, ::-1, :].copy()
            occ_mask  = occ_mask [:, ::-1   ].copy()
            bbox_mask = bbox_mask[:, ::-1   ].copy()
        if np.random.rand() > 0.8:
            occ       = occ      [::-1, :, :].copy()
            clean     = clean    [::-1, :, :].copy()
            occ_mask  = occ_mask [::-1, :   ].copy()
            bbox_mask = bbox_mask[::-1, :   ].copy()
        factor = np.random.uniform(0.8, 1.2)
        occ    = np.clip(occ   * factor, 0.0, 1.0)
        clean  = np.clip(clean * factor, 0.0, 1.0)
        return occ, clean, occ_mask, bbox_mask

    def __getitem__(self, idx):
        s = self.samples[idx]
        occ_img,   _           = self._load_rgb(s['occ_path'])
        clean_img, original_hw = self._load_rgb(s['clean_path'])
        occ_mask_path = s.get('mask_path', '')
        occ_mask      = self._load_occ_mask(occ_mask_path)

        Ht, Wt   = self.target_size[1], self.target_size[0]
        base     = os.path.splitext(os.path.basename(s['clean_path']))[0]
        xml_path = os.path.join(self.annot_dir, base + '.xml')
        bboxes   = parse_annotation(xml_path, original_hw, (Ht, Wt))
        bbox_mask = (np.ones((Ht, Wt), dtype=np.float32) if len(bboxes) == 0
                     else make_bbox_mask(Ht, Wt, bboxes))

        if self.augment:
            occ_img, clean_img, occ_mask, bbox_mask = \
                self._augment(occ_img, clean_img, occ_mask, bbox_mask)

        occ_img   = np.clip(occ_img,   0.0, 1.0)
        clean_img = np.clip(clean_img, 0.0, 1.0)
        occ_mask  = np.clip(occ_mask,  0.0, 1.0)
        bbox_mask = np.clip(bbox_mask, 0.0, 1.0)

        x = np.concatenate([occ_img, occ_mask[..., None], bbox_mask[..., None]],
                           axis=-1).astype(np.float32)
        y = clean_img.astype(np.float32)

        return (
            torch.from_numpy(x.transpose(2, 0, 1)).float(),
            torch.from_numpy(y.transpose(2, 0, 1)).float(),
            torch.from_numpy(occ_mask).unsqueeze(0).float(),
            torch.from_numpy(bbox_mask).unsqueeze(0).float(),
            os.path.basename(s['clean_path']),
        )


def collate_fn(batch):
    xs, ys, oms, bms, fns = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(oms), torch.stack(bms), list(fns)


# ── Partial Convolution (Liu et al., 2018) ──
class PartialConv2d(nn.Conv2d):
    """
    Mask-aware convolution. Output at a location is renormalized by how
    many valid (unmasked) input pixels contributed to it, and the mask
    is updated so a location becomes 'valid' as soon as its receptive
    field touches ANY valid pixel.
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
        self.act = (nn.ReLU(inplace=True) if act == 'relu'
                    else nn.LeakyReLU(0.2, inplace=True) if act == 'leaky'
                    else nn.Identity())

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.act(self.bn(x))
        return x, mask


class PConvUNet(nn.Module):
    """Model H -- Partial-Convolution U-Net. Encoder/decoder with
    PartialConv2d throughout; the mask is a single-channel signal
    (1=valid) broadcast against however many feature channels each
    layer has, propagated through the network and combined with the
    encoder's mask at each skip connection via elementwise max."""
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
        m0 = mask  # single-channel, 1=valid

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
    m = hole_mask
    tv_h = ((pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs() * m[:, :, 1:, :]).mean()
    tv_w = ((pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs() * m[:, :, :, :-1]).mean()
    return tv_h + tv_w

def pconv_loss(pred, gt, occ_mask):
    """Model H -- hole/valid L1 + TV. NOTE: the original paper also adds
    VGG perceptual + style losses; omitted here since they require a
    pretrained VGG that may not be reachable offline. hole/valid L1 + TV
    already captures most of PConv's practical behaviour."""
    hole  = occ_mask.expand_as(pred)
    valid = 1.0 - hole
    n_hole  = hole.sum().clamp(min=1)
    n_valid = valid.sum().clamp(min=1)
    l_hole  = ((pred - gt).abs() * hole).sum() / n_hole
    l_valid = ((pred - gt).abs() * valid).sum() / n_valid
    l_tv    = total_variation_loss(pred, hole)
    return 6.0 * l_hole + 1.0 * l_valid + 0.1 * l_tv


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_history_csv(history, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss'])
        for i, (tr, va) in enumerate(zip(history['train'], history['val']), 1):
            w.writerow([i, float(tr), float(va)])

def save_loss_curve(history, path):
    plt.figure(figsize=(8, 4))
    plt.plot(history['train'], label='train')
    plt.plot(history['val'],   label='val')
    plt.title('Model H — Partial-Conv U-Net (hole/valid L1 + TV)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def train():
    start_time = time.time()
    print("=" * 60)
    print("  MODEL H — Partial-Conv U-Net")
    print("  Input : 5ch (RGB + occ_mask + bbox_mask)")
    print("  Loss  : hole L1 + valid L1 + TV")
    print("=" * 60)

    train_samples, val_samples, split_obj = load_existing_split(BASE_SPLIT_PATH)
    shutil.copy2(BASE_SPLIT_PATH, SPLIT_COPY_PATH)

    train_ds = PConvDataset(train_samples, ANNOT_DIR, MASK_DIR, TARGET_SIZE, augment=True)
    val_ds   = PConvDataset(val_samples,   ANNOT_DIR, MASK_DIR, TARGET_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=False)

    model     = PConvUNet(in_ch=5, out_ch=3).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    save_json(CONFIG_JSON_PATH, {
        'model': 'Model H — Partial-Conv U-Net', 'in_channels': 5,
        'loss': '6.0*hole_L1 + 1.0*valid_L1 + 0.1*TV', 'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS, 'LR': LR, 'EARLY_STOP': EARLY_STOP,
        'TARGET_SIZE': list(TARGET_SIZE), 'DEVICE': str(DEVICE),
        'n_train': len(train_samples), 'n_val': len(val_samples), 'n_params': n_params,
    })

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, nan_batches = 0.0, 0
        for x, y, occ_mask, bbox_mask, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            occ_mask, bbox_mask = occ_mask.to(DEVICE), bbox_mask.to(DEVICE)
            valid_mask = 1.0 - occ_mask

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, valid_mask)
            loss = pconv_loss(pred, y, occ_mask)

            if not torch.isfinite(loss):
                nan_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        n_valid = max(1, len(train_loader) - nan_batches)
        train_loss /= n_valid
        if nan_batches > 0:
            print(f"  [WARN] {nan_batches} NaN batches skipped")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, occ_mask, bbox_mask, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                occ_mask, bbox_mask = occ_mask.to(DEVICE), bbox_mask.to(DEVICE)
                valid_mask = 1.0 - occ_mask
                pred = model(x, valid_mask)
                loss = pconv_loss(pred, y, occ_mask)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history['train'].append(float(train_loss))
        history['val'].append(float(val_loss))
        save_json(HISTORY_JSON_PATH, history)
        save_history_csv(history, HISTORY_CSV_PATH)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_epoch, patience_counter = float(val_loss), epoch + 1, 0
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val': best_val, 'history': history,
                'in_ch': 5, 'arch': 'pconv', 'model_name': 'Model H — Partial-Conv U-Net',
            }, BEST_MODEL_PATH)
            print(f"  Saved best model -> {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    torch.save({
        'model_state_dict': model.state_dict(), 'history': history,
        'in_ch': 5, 'arch': 'pconv', 'model_name': 'Model H — Partial-Conv U-Net',
    }, FINAL_MODEL_PATH)

    save_loss_curve(history, LOSS_PLOT_PATH)
    end_time = time.time()

    save_json(SUMMARY_JSON_PATH, {
        'model': 'Model H — Partial-Conv U-Net', 'best_val_loss': best_val,
        'best_epoch': best_epoch, 'epochs_completed': len(history['train']),
        'duration_seconds': end_time - start_time,
        'best_model_path': BEST_MODEL_PATH, 'final_model_path': FINAL_MODEL_PATH,
        'n_params': n_params,
    })

    print(f"\nTraining complete — Model H")
    print(f"  Best val loss : {best_val:.6f} at epoch {best_epoch}")
    print(f"  Best model    : {BEST_MODEL_PATH}")
    print(f"  Final model   : {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    train()
