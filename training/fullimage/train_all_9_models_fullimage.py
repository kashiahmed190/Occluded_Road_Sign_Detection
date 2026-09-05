"""
train_all_9_models_fullimage.py
=================================
Trains ALL 9 inpainting models sequentially, on the FULL (877) images
with REAL PASCAL VOC XML bounding-box annotations -- not the 1244
pre-cropped patches used in the earlier version of this pipeline.

  Model A — Base U-Net         3ch  | plain MSE
  Model B — Mask U-Net         5ch  | plain MSE
  Model C — ROI U-Net          5ch  | 3-region weighted MSE (real bbox)
  Model D — Enhanced U-Net     5ch  | region MSE + SSIM + L1
  Model E — Advanced U-Net     5ch  | residual+ASPP+attention, composite loss
  Model F — Plain Autoencoder  5ch  | plain MSE, NO skip connections
  Model G — Context-Enc. GAN   5ch  | region L2 + adversarial (G/D)
  Model H — Partial-Conv UNet  5ch  | hole/valid L1 + TV
  Model I — ViT Inpainter      5ch  | region MSE + L1, pure transformer

Folder layout expected (same as every other script in this pipeline):
    d/            clean full images
    occ/          occluded full images
    occ_masks/    occlusion masks
    e/            PASCAL VOC XML annotations (one per clean image, same basename)
    traintestsplit/data_split.json

Each model is saved in its own folder (matching evaluate_all_models.py's
expected paths exactly), using the SAME train/test split for all 9.

Usage:
    python train_all_9_models_fullimage.py
"""

import os, json, csv, time, shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

OCCLUDED_DIR = "occ"
MASK_DIR     = "occ_masks"
CLEAN_DIR    = "d"
ANNOT_DIR    = "e"

BASE_SPLIT_PATH = os.path.join("traintestsplit", "data_split.json")
TARGET_SIZE  = (256, 256)
BATCH_SIZE   = 8
EPOCHS       = 60
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 12
NUM_WORKERS  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_HIDDEN, W_VISIBLE, W_BG = 1.0, 0.4, 0.1
LAMBDA_L2, LAMBDA_ADV = 100.0, 1.0

print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════
# SPLIT + XML ANNOTATION HELPERS (shared by every model)
# ═══════════════════════════════════════════════════════════════

def load_existing_split(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path) as f:
        split = json.load(f)
    train = split["train"]
    test  = split.get("test", split.get("val", []))
    print(f"Split loaded: train={len(train)}  val={len(test)}")
    return train, test

def parse_annotation(xml_path, original_hw, target_hw):
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        H0, W0 = original_hw
        Ht, Wt = target_hw
        sx, sy = Wt / float(W0), Ht / float(H0)
        bboxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = int(float(bb.find("xmin").text) * sx)
            ymin = int(float(bb.find("ymin").text) * sy)
            xmax = int(float(bb.find("xmax").text) * sx)
            ymax = int(float(bb.find("ymax").text) * sy)
            xmin = max(0, min(Wt-1, xmin)); xmax = max(0, min(Wt, xmax))
            ymin = max(0, min(Ht-1, ymin)); ymax = max(0, min(Ht, ymax))
            if xmax > xmin and ymax > ymin:
                bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")
        return []

def make_bbox_mask(h, w, bboxes):
    m = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in bboxes:
        m[y1:y2, x1:x2] = 1.0
    return m


# ═══════════════════════════════════════════════════════════════
# SHARED DATASET  (always returns 5ch x; 3ch models slice x[:, :3])
# ═══════════════════════════════════════════════════════════════

class InpaintDataset(Dataset):
    def __init__(self, samples, annot_dir=ANNOT_DIR, target_size=TARGET_SIZE, augment=False):
        self.samples     = samples
        self.annot_dir   = annot_dir
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
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        if mask is None:
            Ht, Wt = self.target_size[1], self.target_size[0]
            return np.zeros((Ht, Wt), dtype=np.float32)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)

    def _augment(self, occ, clean, occ_mask, bbox_mask):
        if np.random.rand() > 0.5:
            occ, clean = occ[:, ::-1, :].copy(), clean[:, ::-1, :].copy()
            occ_mask, bbox_mask = occ_mask[:, ::-1].copy(), bbox_mask[:, ::-1].copy()
        if np.random.rand() > 0.8:
            occ, clean = occ[::-1, :, :].copy(), clean[::-1, :, :].copy()
            occ_mask, bbox_mask = occ_mask[::-1, :].copy(), bbox_mask[::-1, :].copy()
        factor = np.random.uniform(0.8, 1.2)
        occ, clean = np.clip(occ * factor, 0, 1), np.clip(clean * factor, 0, 1)
        return occ, clean, occ_mask, bbox_mask

    def __getitem__(self, idx):
        s = self.samples[idx]
        occ_img,   _           = self._load_rgb(s['occ_path'])
        clean_img, original_hw = self._load_rgb(s['clean_path'])
        occ_mask = self._load_occ_mask(s.get('mask_path', ''))

        Ht, Wt   = self.target_size[1], self.target_size[0]
        base     = os.path.splitext(os.path.basename(s['clean_path']))[0]
        xml_path = os.path.join(self.annot_dir, base + '.xml')
        bboxes   = parse_annotation(xml_path, original_hw, (Ht, Wt))
        bbox_mask = np.ones((Ht, Wt), dtype=np.float32) if not bboxes else make_bbox_mask(Ht, Wt, bboxes)

        if self.augment:
            occ_img, clean_img, occ_mask, bbox_mask = self._augment(occ_img, clean_img, occ_mask, bbox_mask)

        occ_img, clean_img = np.clip(occ_img, 0, 1), np.clip(clean_img, 0, 1)
        occ_mask, bbox_mask = np.clip(occ_mask, 0, 1), np.clip(bbox_mask, 0, 1)

        x = np.concatenate([occ_img, occ_mask[..., None], bbox_mask[..., None]], axis=-1).astype(np.float32)
        y = clean_img.astype(np.float32)

        return (
            torch.from_numpy(x.transpose(2, 0, 1)).float(),
            torch.from_numpy(y.transpose(2, 0, 1)).float(),
            torch.from_numpy(occ_mask).unsqueeze(0).float(),
            torch.from_numpy(bbox_mask).unsqueeze(0).float(),
        )

def collate_fn(batch):
    xs, ys, oms, bms = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(oms), torch.stack(bms)


# ═══════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.drop(x)

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.proj  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        res = self.proj(x)
        x   = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x   = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.drop(F.relu(x + res, inplace=True))


# ── Models A-D: plain U-Net (in_ch 3 or 5) ──────────────────────
class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        self.c1 = ConvBlock(in_ch, 64,   0.1); self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(64,   128,   0.1); self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(128,  256,   0.2); self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(256,  512,   0.2); self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(512,  1024,  0.3)
        self.u6 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.c6 = ConvBlock(1024, 512, 0.2)
        self.u7 = nn.ConvTranspose2d(512,  256, 2, stride=2); self.c7 = ConvBlock(512,  256, 0.2)
        self.u8 = nn.ConvTranspose2d(256,  128, 2, stride=2); self.c8 = ConvBlock(256,  128, 0.1)
        self.u9 = nn.ConvTranspose2d(128,   64, 2, stride=2); self.c9 = ConvBlock(128,   64, 0.1)
        self.out = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        c1 = self.c1(x);  p1 = self.p1(c1)
        c2 = self.c2(p1); p2 = self.p2(c2)
        c3 = self.c3(p2); p3 = self.p3(c3)
        c4 = self.c4(p3); p4 = self.p4(c4)
        c5 = self.c5(p4)
        c6 = self.c6(torch.cat([self.u6(c5), c4], 1))
        c7 = self.c7(torch.cat([self.u7(c6), c3], 1))
        c8 = self.c8(torch.cat([self.u8(c7), c2], 1))
        c9 = self.c9(torch.cat([self.u9(c8), c1], 1))
        return torch.sigmoid(self.out(c9))


# ── Model E: residual + ASPP + attention gates + deep supervision ──
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.Wx = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        att = F.relu(self.Wg(g) + self.Wx(x), inplace=True)
        return x * self.psi(att)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.a1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.a2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.a3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.a4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.global_avg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(out_ch*5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.3))
    def forward(self, x):
        h, w = x.shape[-2:]
        gap = F.interpolate(self.global_avg(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.proj(torch.cat([self.a1(x), self.a2(x), self.a3(x), self.a4(x), gap], dim=1))

class AdvancedUNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        self.e1 = ResConvBlock(in_ch, 64,   0.1); self.p1 = nn.MaxPool2d(2)
        self.e2 = ResConvBlock(64,   128,   0.1); self.p2 = nn.MaxPool2d(2)
        self.e3 = ResConvBlock(128,  256,   0.2); self.p3 = nn.MaxPool2d(2)
        self.e4 = ResConvBlock(256,  512,   0.2); self.p4 = nn.MaxPool2d(2)
        self.bottleneck = ASPP(512, 1024)
        self.ag4 = AttentionGate(1024, 512, 256)
        self.ag3 = AttentionGate(512,  256, 128)
        self.ag2 = AttentionGate(256,  128, 64)
        self.ag1 = AttentionGate(128,  64,  32)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.d4 = ResConvBlock(1024, 512, 0.2)
        self.u3 = nn.ConvTranspose2d(512,  256, 2, stride=2); self.d3 = ResConvBlock(512,  256, 0.2)
        self.u2 = nn.ConvTranspose2d(256,  128, 2, stride=2); self.d2 = ResConvBlock(256,  128, 0.1)
        self.u1 = nn.ConvTranspose2d(128,   64, 2, stride=2); self.d1 = ResConvBlock(128,   64, 0.1)
        self.out_main = nn.Conv2d(64, out_ch, 1)
        self.out_aux  = nn.Conv2d(256, out_ch, 1)
    def forward(self, x, return_aux=False):
        e1 = self.e1(x);  p1 = self.p1(e1)
        e2 = self.e2(p1); p2 = self.p2(e2)
        e3 = self.e3(p2); p3 = self.p3(e3)
        e4 = self.e4(p3); p4 = self.p4(e4)
        b = self.bottleneck(p4)
        d4 = self.d4(torch.cat([self.u4(b),  self.ag4(b,  e4)], 1))
        d3 = self.d3(torch.cat([self.u3(d4), self.ag3(d4, e3)], 1))
        d2 = self.d2(torch.cat([self.u2(d3), self.ag2(d3, e2)], 1))
        d1 = self.d1(torch.cat([self.u1(d2), self.ag1(d2, e1)], 1))
        out = torch.sigmoid(self.out_main(d1))
        if return_aux:
            return out, torch.sigmoid(self.out_aux(d3))
        return out


# ── Model F: plain autoencoder, no skip connections ─────────────
class PlainAutoencoder(nn.Module):
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
        x = self.p1(self.e1(x)); x = self.p2(self.e2(x))
        x = self.p3(self.e3(x)); x = self.p4(self.e4(x))
        x = self.bottleneck(x)
        x = self.d4(x); x = self.d3(x); x = self.d2(x); x = self.d1(x)
        return torch.sigmoid(self.out(x))


# ── Model G: Context-Encoder GAN ────────────────────────────────
class CEGenerator(nn.Module):
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        def down(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(co), nn.LeakyReLU(0.2, inplace=True))
        def up(ci, co):
            return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.enc = nn.Sequential(down(in_ch, 64), down(64, 128), down(128, 256),
                                  down(256, 512), down(512, 512), down(512, 512))
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(up(512, 512), up(512, 512), up(512, 256), up(256, 128),
                                 up(128, 64), nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1))
    def forward(self, x):
        z = self.bottleneck(self.enc(x))
        return torch.sigmoid(self.dec(z))

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        def block(ci, co, norm=True, stride=2):
            layers = [nn.Conv2d(ci, co, 4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.net = nn.Sequential(*block(in_ch, 64, norm=False), *block(64, 128), *block(128, 256),
                                 *block(256, 512, stride=1), nn.Conv2d(512, 1, 4, stride=1, padding=1))
    def forward(self, x):
        return self.net(x)


# ── Model H: Partial-Conv U-Net ─────────────────────────────────
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('weight_mask_updater', torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.slide_win_size = self.kernel_size[0] * self.kernel_size[1]
    def forward(self, x, mask_in):
        with torch.no_grad():
            update_mask = F.conv2d(mask_in, self.weight_mask_updater.to(x.dtype),
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
                    else nn.LeakyReLU(0.2, inplace=True) if act == 'leaky' else nn.Identity())
    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        return self.act(self.bn(x)), mask

class PConvUNet(nn.Module):
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
        m0 = mask
        x1, m1 = self.e1(x, m0); x2, m2 = self.e2(x1, m1); x3, m3 = self.e3(x2, m2)
        x4, m4 = self.e4(x3, m3); x5, m5 = self.e5(x4, m4)
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


# ── Model I: ViT Inpainter ───────────────────────────────────────
class ViTInpainter(nn.Module):
    def __init__(self, in_ch=5, out_ch=3, img_size=256, patch=16, embed_dim=384, depth=6, n_heads=6, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        self.patch, self.grid, self.out_ch = patch, img_size // patch, out_ch
        self.n_patches = self.grid * self.grid
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=int(embed_dim*mlp_ratio),
                                           dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch * patch * out_ch)
    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        tokens = self.norm(self.transformer(tokens))
        pix = self.head(tokens).view(B, self.grid, self.grid, self.patch, self.patch, self.out_ch)
        pix = pix.permute(0, 5, 1, 3, 2, 4).contiguous()
        return torch.sigmoid(pix.view(B, self.out_ch, self.grid*self.patch, self.grid*self.patch))


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _gauss_win(ws=11, sigma=1.5, C=3, device='cpu'):
    coords = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g = torch.exp(-(coords**2)/(2*sigma**2)); g = g/g.sum()
    w = g.view(1,1,1,ws).transpose(2,3) @ g.view(1,1,1,ws)
    return w.expand(C,1,ws,ws).contiguous()

def ssim_loss(x, y, ws=11, eps=1e-8):
    x, y = x.clamp(0,1), y.clamp(0,1)
    C = x.shape[1]; win = _gauss_win(ws, 1.5, C, x.device); pad = ws // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=C); mu_y = F.conv2d(y, win, padding=pad, groups=C)
    sg_x2 = (F.conv2d(x*x, win, padding=pad, groups=C) - mu_x**2).clamp(min=0)
    sg_y2 = (F.conv2d(y*y, win, padding=pad, groups=C) - mu_y**2).clamp(min=0)
    sg_xy = F.conv2d(x*y, win, padding=pad, groups=C) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_x*mu_y+c1)*(2*sg_xy+c2)) / ((mu_x**2+mu_y**2+c1)*(sg_x2+sg_y2+c2)+eps)
    return 1.0 - ssim.mean()

def edge_loss(pred, gt):
    def sobel(x):
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device)
        ky = kx.t()
        kx = kx.view(1,1,3,3).expand(x.shape[1],-1,-1,-1); ky = ky.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)
        gx = F.conv2d(x, kx, padding=1, groups=x.shape[1]); gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
        return torch.sqrt(gx**2 + gy**2 + 1e-8)
    return F.l1_loss(sobel(pred), sobel(gt))

def region_weights(occ_mask, bbox_mask):
    """Uses the REAL XML-derived bbox_mask: hidden=occluded&inbbox,
    visible=inbbox&notoccluded, background=outside bbox."""
    occ, bbox = occ_mask.float(), bbox_mask.float()
    return W_HIDDEN*(bbox*occ) + W_VISIBLE*(bbox*(1.0-occ)) + W_BG*(1.0-bbox)

def total_variation_loss(pred, hole_mask):
    m = hole_mask
    tv_h = ((pred[:,:,1:,:] - pred[:,:,:-1,:]).abs() * m[:,:,1:,:]).mean()
    tv_w = ((pred[:,:,:,1:] - pred[:,:,:,:-1]).abs() * m[:,:,:,:-1]).mean()
    return tv_h + tv_w

def loss_A(pred, gt, occ_mask, bbox_mask):
    return F.mse_loss(pred, gt)

def loss_B(pred, gt, occ_mask, bbox_mask):
    return F.mse_loss(pred, gt)

def loss_C(pred, gt, occ_mask, bbox_mask):
    w = region_weights(occ_mask, bbox_mask).expand_as(pred)
    return (w * (pred - gt)**2).mean()

def loss_D(pred, gt, occ_mask, bbox_mask):
    w = region_weights(occ_mask, bbox_mask).expand_as(pred)
    mse = (w * (pred - gt)**2).mean()
    ss = ssim_loss(pred, gt)
    hidden = (occ_mask * bbox_mask).expand_as(pred)
    n = hidden.sum().clamp(min=1)
    l1 = ((pred - gt).abs() * hidden).sum() / n
    return 0.5*mse + 0.3*ss + 0.2*l1

def loss_E(pred, gt, occ_mask, bbox_mask, aux=None):
    w = region_weights(occ_mask, bbox_mask).expand_as(pred)
    mse = (w * (pred - gt)**2).mean()
    ss = ssim_loss(pred, gt)
    hidden = (occ_mask * bbox_mask).expand_as(pred)
    n = hidden.sum().clamp(min=1)
    l1 = ((pred - gt).abs() * hidden).sum() / n
    el = edge_loss(pred, gt)
    loss = 0.40*mse + 0.25*ss + 0.20*l1 + 0.10*el
    if aux is not None:
        gt_ds = F.interpolate(gt, size=aux.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + 0.05 * F.mse_loss(aux, gt_ds)
    return loss

def loss_F(pred, gt, occ_mask, bbox_mask):
    return F.mse_loss(pred, gt)

def loss_G_recon(pred, gt, occ_mask, bbox_mask):
    w = region_weights(occ_mask, bbox_mask).expand_as(pred)
    return (w * (pred - gt)**2).mean()

def loss_H(pred, gt, occ_mask, bbox_mask):
    hole = occ_mask.expand_as(pred); valid = 1.0 - hole
    n_hole = hole.sum().clamp(min=1); n_valid = valid.sum().clamp(min=1)
    l_hole = ((pred - gt).abs() * hole).sum() / n_hole
    l_valid = ((pred - gt).abs() * valid).sum() / n_valid
    l_tv = total_variation_loss(pred, hole)
    return 6.0*l_hole + 1.0*l_valid + 0.1*l_tv

def loss_I(pred, gt, occ_mask, bbox_mask):
    w = region_weights(occ_mask, bbox_mask).expand_as(pred)
    mse = (w * (pred - gt)**2).mean()
    hidden = (occ_mask * bbox_mask).expand_as(pred)
    n = hidden.sum().clamp(min=1)
    l1 = ((pred - gt).abs() * hidden).sum() / n
    return 0.7*mse + 0.3*l1


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
        for i in range(len(history[cols[0]])):
            w.writerow([i+1] + [f"{history[c][i]:.6f}" for c in cols])

def save_curve(history, path, model_name):
    plt.figure(figsize=(8, 4))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.title(f'{model_name} — loss curve')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY  (paths match evaluate_all_models.py exactly)
# ═══════════════════════════════════════════════════════════════

MODELS = [
    {"name": "Model A — Base U-Net",        "arch": "unet",     "in_ch": 3, "loss_fn": loss_A,
     "save_dir": "inpainting_model",              "ckpt": "best_inpainting.pth"},
    {"name": "Model B — Mask U-Net",        "arch": "unet",     "in_ch": 5, "loss_fn": loss_B,
     "save_dir": "model_B_mask_unet",              "ckpt": "best_mask_unet.pth"},
    {"name": "Model C — ROI U-Net",         "arch": "unet",     "in_ch": 5, "loss_fn": loss_C,
     "save_dir": "model_C_roi_unet",               "ckpt": "best_roi_unet.pth"},
    {"name": "Model D — Enhanced U-Net",    "arch": "unet",     "in_ch": 5, "loss_fn": loss_D,
     "save_dir": "enhanced_unetapril2026",         "ckpt": "best_roi_unet_ssim.pth"},
    {"name": "Model E — Advanced U-Net",    "arch": "advanced", "in_ch": 5, "loss_fn": loss_E,
     "save_dir": "model_E_advanced_unet",          "ckpt": "best_advanced_unet.pth"},
    {"name": "Model F — Plain Autoencoder", "arch": "plain_ae", "in_ch": 5, "loss_fn": loss_F,
     "save_dir": "model_F_plain_autoencoder",      "ckpt": "best_model_F.pth"},
    {"name": "Model G — Context-Encoder GAN", "arch": "gan",    "in_ch": 5, "loss_fn": None,
     "save_dir": "model_G_context_encoder_gan",    "ckpt": "best_model_G.pth"},
    {"name": "Model H — Partial-Conv U-Net", "arch": "pconv",   "in_ch": 5, "loss_fn": loss_H,
     "save_dir": "model_H_partial_conv_unet",      "ckpt": "best_model_H.pth"},
    {"name": "Model I — ViT Inpainter",     "arch": "vit",      "in_ch": 5, "loss_fn": loss_I,
     "save_dir": "model_I_vit_inpainter",          "ckpt": "best_model_I.pth"},
]


# ═══════════════════════════════════════════════════════════════
# TRAIN — single-network models (A,B,C,D,E,F,H,I)
# ═══════════════════════════════════════════════════════════════

def build_model(cfg):
    arch, in_ch = cfg['arch'], cfg['in_ch']
    if arch == 'unet':
        return UNet(in_ch=in_ch, out_ch=3).to(DEVICE)
    if arch == 'advanced':
        return AdvancedUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
    if arch == 'plain_ae':
        return PlainAutoencoder(in_ch=in_ch, out_ch=3).to(DEVICE)
    if arch == 'pconv':
        return PConvUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
    if arch == 'vit':
        return ViTInpainter(in_ch=in_ch, out_ch=3, img_size=TARGET_SIZE[0]).to(DEVICE)
    raise ValueError(f"Unknown arch {arch}")


def train_single_network(cfg, train_loader, val_loader, save_dir):
    arch, in_ch, loss_fn = cfg['arch'], cfg['in_ch'], cfg['loss_fn']
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train': [], 'val': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, skipped = 0.0, 0
        for x, y, occ_mask, bbox_mask in train_loader:
            xin = x[:, :in_ch].to(DEVICE)
            y, occ_mask, bbox_mask = y.to(DEVICE), occ_mask.to(DEVICE), bbox_mask.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if arch == 'pconv':
                valid_mask = 1.0 - occ_mask
                pred = model(xin, valid_mask)
                loss = loss_fn(pred, y, occ_mask, bbox_mask)
            elif arch == 'advanced':
                pred, aux = model(xin, return_aux=True)
                loss = loss_fn(pred, y, occ_mask, bbox_mask, aux=aux)
            else:
                pred = model(xin)
                loss = loss_fn(pred, y, occ_mask, bbox_mask)

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
            for x, y, occ_mask, bbox_mask in val_loader:
                xin = x[:, :in_ch].to(DEVICE)
                y, occ_mask, bbox_mask = y.to(DEVICE), occ_mask.to(DEVICE), bbox_mask.to(DEVICE)
                if arch == 'pconv':
                    valid_mask = 1.0 - occ_mask
                    pred = model(xin, valid_mask)
                elif arch == 'advanced':
                    pred = model(xin, return_aux=False)
                else:
                    pred = model(xin)
                loss = loss_fn(pred, y, occ_mask, bbox_mask)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history['train'].append(float(train_loss))
        history['val'].append(float(val_loss))
        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:03d}/{EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, best_epoch, patience_counter = val_loss, epoch + 1, 0
            torch.save({'epoch': epoch+1, 'model_name': cfg['name'], 'arch': arch, 'in_ch': in_ch,
                        'model_state_dict': model.state_dict(), 'best_val': best_val, 'history': history},
                       os.path.join(save_dir, cfg['ckpt']))
            print(f"    Best model saved (val={best_val:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"  Early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    torch.save({'model_name': cfg['name'], 'arch': arch, 'in_ch': in_ch,
                'model_state_dict': model.state_dict(), 'history': history},
               os.path.join(save_dir, cfg['ckpt'].replace('best', 'final')))
    return history, best_val, best_epoch, elapsed, n_params


# ═══════════════════════════════════════════════════════════════
# TRAIN — Model G (GAN, alternating G/D updates)
# ═══════════════════════════════════════════════════════════════

def train_gan(cfg, train_loader, val_loader, save_dir):
    G = CEGenerator(in_ch=cfg['in_ch'], out_ch=3).to(DEVICE)
    D = PatchDiscriminator(in_ch=3).to(DEVICE)
    n_params = (sum(p.numel() for p in G.parameters() if p.requires_grad) +
                sum(p.numel() for p in D.parameters() if p.requires_grad))
    print(f"  Parameters (G+D): {n_params:,}")

    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=6, min_lr=1e-6)
    bce = nn.BCEWithLogitsLoss()

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train_g': [], 'train_d': [], 'val': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        G.train(); D.train()
        g_sum, d_sum, skipped = 0.0, 0.0, 0
        for x, y, occ_mask, bbox_mask in train_loader:
            xin = x[:, :cfg['in_ch']].to(DEVICE)
            y, occ_mask, bbox_mask = y.to(DEVICE), occ_mask.to(DEVICE), bbox_mask.to(DEVICE)

            with torch.no_grad():
                fake = G(xin)
            real_label = torch.ones(D(y).shape, device=DEVICE)
            fake_label = torch.zeros_like(real_label)

            opt_d.zero_grad(set_to_none=True)
            d_loss = 0.5 * (bce(D(y), real_label) + bce(D(fake.detach()), fake_label))
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake = G(xin)
            adv_loss = bce(D(fake), real_label)
            recon_loss = loss_G_recon(fake, y, occ_mask, bbox_mask)
            g_loss = LAMBDA_L2 * recon_loss + LAMBDA_ADV * adv_loss

            if not torch.isfinite(g_loss):
                skipped += 1
                continue
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()

            g_sum += g_loss.item(); d_sum += d_loss.item()

        n_valid = max(1, len(train_loader) - skipped)
        train_g, train_d = g_sum / n_valid, d_sum / n_valid

        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, occ_mask, bbox_mask in val_loader:
                xin = x[:, :cfg['in_ch']].to(DEVICE)
                y, occ_mask, bbox_mask = y.to(DEVICE), occ_mask.to(DEVICE), bbox_mask.to(DEVICE)
                pred = G(xin)
                loss = loss_G_recon(pred, y, occ_mask, bbox_mask)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history['train_g'].append(float(train_g))
        history['train_d'].append(float(train_d))
        history['val'].append(float(val_loss))
        sched_g.step(val_loss)

        print(f"  Epoch {epoch+1:03d}/{EPOCHS}  G={train_g:.5f}  D={train_d:.5f}  val(recon)={val_loss:.5f}")

        if val_loss < best_val:
            best_val, best_epoch, patience_counter = val_loss, epoch + 1, 0
            torch.save({'model_name': cfg['name'], 'arch': cfg['arch'], 'in_ch': cfg['in_ch'],
                        'generator_state_dict': G.state_dict(), 'discriminator_state_dict': D.state_dict(),
                        'best_val': best_val, 'history': history}, os.path.join(save_dir, cfg['ckpt']))
            print(f"    Best model saved (val={best_val:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"  Early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    torch.save({'model_name': cfg['name'], 'arch': cfg['arch'], 'in_ch': cfg['in_ch'],
                'generator_state_dict': G.state_dict(), 'discriminator_state_dict': D.state_dict(),
                'history': history}, os.path.join(save_dir, cfg['ckpt'].replace('best', 'final')))
    return history, best_val, best_epoch, elapsed, n_params


# ═══════════════════════════════════════════════════════════════
# TRAIN ONE MODEL (dispatcher)
# ═══════════════════════════════════════════════════════════════

def train_model(cfg, train_samples, val_samples):
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Training : {cfg['name']}")
    print(f"  Arch     : {cfg['arch'].upper()}   Input ch: {cfg['in_ch']}")
    print(f"  Save dir : {save_dir}/")
    print(f"{'='*64}")

    train_ds = InpaintDataset(train_samples, ANNOT_DIR, TARGET_SIZE, augment=True)
    val_ds   = InpaintDataset(val_samples,   ANNOT_DIR, TARGET_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    if cfg['arch'] == 'gan':
        history, best_val, best_epoch, elapsed, n_params = train_gan(cfg, train_loader, val_loader, save_dir)
    else:
        history, best_val, best_epoch, elapsed, n_params = train_single_network(cfg, train_loader, val_loader, save_dir)

    save_json(os.path.join(save_dir, 'history.json'), history)
    save_csv(history, os.path.join(save_dir, 'epoch_log.csv'))
    save_curve(history, os.path.join(save_dir, 'loss_curve.png'), cfg['name'])
    save_json(os.path.join(save_dir, 'training_summary.json'), {
        'model_name': cfg['name'], 'arch': cfg['arch'], 'in_ch': cfg['in_ch'],
        'best_val_loss': best_val, 'best_epoch': best_epoch,
        'epochs_completed': len(history.get('val', [])), 'duration_seconds': elapsed,
        'parameters': n_params,
    })

    print(f"\n  Finished {cfg['name']}")
    print(f"  Best val : {best_val:.5f} at epoch {best_epoch}")
    print(f"  Time     : {elapsed/60:.1f} min")
    print(f"  Saved to : {save_dir}/")
    return best_val, best_epoch


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading split...")
    train_samples, val_samples = load_existing_split(BASE_SPLIT_PATH)

    results = []
    for cfg in MODELS:
        best_val, best_epoch = train_model(cfg, train_samples, val_samples)
        results.append({'model': cfg['name'], 'best_val': best_val,
                        'best_epoch': best_epoch, 'save_dir': cfg['save_dir']})

    print(f"\n{'='*64}")
    print("  ALL 9 MODELS TRAINING COMPLETE")
    print(f"{'='*64}")
    print(f"  {'Model':<32} {'Best Val':>10} {'Epoch':>7}")
    print("  " + "-" * 54)
    for r in results:
        print(f"  {r['model']:<32} {r['best_val']:>10.5f} {r['best_epoch']:>7}")

    save_json("training_all_9_results.json", results)
    print("\n  Summary saved -> training_all_9_results.json")
    print("\nNext step: run evaluate_all_models.py to compare all 9 on the test split.")


if __name__ == "__main__":
    main()
