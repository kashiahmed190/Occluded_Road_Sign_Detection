"""
evaluate_all_models.py
======================
Evaluates all 5 models on the same test split:
  Model A -- Base U-Net        (3ch input, inpainting_model/best_inpainting.pth)
  Model B -- Mask U-Net        (5ch input, model_B_mask_unet/best_mask_unet.pth)
  Model C -- ROI U-Net         (5ch input, model_C_roi_unet/best_roi_unet.pth)
  Model D -- Enhanced U-Net    (5ch input, enhanced_unetapril2026/best_roi_unet_ssim.pth)
  Model E -- Advanced U-Net    (5ch input, model_E_advanced_unet/best_advanced_unet.pth)

Metrics per occlusion level (10-80%):
  Full Image  : PSNR, SSIM, MS-SSIM, LPIPS, MSE, MAE, RMSE
  Masked ROI  : PSNR_mask, SSIM_mask, MAE_mask
  Timing      : Total inference (s), Avg ms/image
Output (all saved to  sept2026fullimage/):
  ModelA_Base_UNet.xlsx
  ModelB_Mask_UNet.xlsx
  ModelC_ROI_UNet.xlsx
  ModelD_Enhanced_UNet.xlsx
  ModelE_Advanced_UNet.xlsx
  comparison_all_models.xlsx       <-- all 5 side by side
  all_results.json                 <-- raw numbers backup
Usage:
  python evaluate_all_models.py
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
# ── optional packages ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False
try:
    import lpips as lpips_lib
    _lpips_fn   = lpips_lib.LPIPS(net='vgg')   # vgg is more reliable than alex
    _HAVE_LPIPS = True
    print("[OK]   lpips loaded (VGG)")
except ImportError:
    print("[INFO] lpips not found -- attempting auto-install...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "lpips", "-q"],
                   check=False)
    try:
        import lpips as lpips_lib
        _lpips_fn   = lpips_lib.LPIPS(net='vgg')
        _HAVE_LPIPS = True
        print("[OK]   lpips installed and loaded (VGG)")
    except Exception as e2:
        _lpips_fn   = None
        _HAVE_LPIPS = False
        print(f"[WARN] lpips still unavailable ({e2}) -- using Sobel gradient proxy")
except Exception as e:
    _lpips_fn   = None
    _HAVE_LPIPS = False
    print(f"[WARN] lpips error ({e}) -- using Sobel gradient proxy")
try:
    from pytorch_msssim import ms_ssim as _ms_ssim_fn
    _HAVE_MSSSIM = True
    print("[OK]   pytorch_msssim loaded")
except Exception:
    _HAVE_MSSSIM = False
    print("[INFO] pytorch_msssim not installed -- using built-in MS-SSIM implementation")
try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    _HAVE_XLSX = True
    print("[OK]   openpyxl loaded")
except Exception:
    _HAVE_XLSX = False
    print("[WARN] openpyxl unavailable -- Excel output disabled")
print()
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SPLIT_PATH  = os.path.join("traintestsplit", "data_split.json")
ANNOT_DIR   = "e"
MASK_DIR    = "occ_masks"
TARGET_SIZE = (256, 256)
BATCH_SIZE  = 8
NUM_WORKERS = 0        # must be 0 when DataLoader is recreated per level inside a loop
OCC_LEVELS  = [10, 20, 30, 40, 50, 60, 70, 80]
OUT_DIR     = "outsept26"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)
# ── model registry ─────────────────────────────────────────────────────────────
#  Each entry: (display_name, checkpoint_path, in_channels, arch)
#  arch: 'unet' (Models A-D, plain encoder-decoder) or 'advanced' (Model E,
#  residual blocks + ASPP + attention gates -- different class entirely).
#  Any model whose checkpoint file does not exist is skipped automatically.
MODELS = [
    (
        "Model A -- Base U-Net",
        os.path.join("inpainting_model", "best_inpainting.pth"),
        3, 'unet',
    ),
    (
        "Model B -- Mask U-Net",
        os.path.join("model_B_mask_unet", "best_mask_unet.pth"),
        5, 'unet',
    ),
    (
        "Model C -- ROI U-Net",
        os.path.join("model_C_roi_unet", "best_roi_unet.pth"),
        5, 'unet',
    ),
    (
        "Model D -- Enhanced U-Net",
        os.path.join("enhanced_unetapril2026", "best_roi_unet_ssim.pth"),
        5, 'unet',
    ),
    (
        "Model E -- Advanced U-Net",
        os.path.join("model_E_advanced_unet", "best_advanced_unet.pth"),
        5, 'advanced',
    ),
    (
        "Model F -- Plain Autoencoder",
        os.path.join("model_F_plain_autoencoder", "best_model_F.pth"),
        5, 'plain_ae',
    ),
    (
        "Model G -- Context-Encoder GAN",
        os.path.join("model_G_context_encoder_gan", "best_model_G.pth"),
        5, 'gan',
    ),
    (
        "Model H -- Partial-Conv U-Net",
        os.path.join("model_H_partial_conv_unet", "best_model_H.pth"),
        5, 'pconv',
    ),
    (
        "Model I -- ViT Inpainter",
        os.path.join("model_I_vit_inpainter", "best_model_I.pth"),
        5, 'vit',
    ),
]
print(f"Device  : {DEVICE}")
print(f"Out dir : {OUT_DIR}/")
print(f"LPIPS   : {'VGG (real LPIPS)' if _HAVE_LPIPS else 'Sobel proxy'}")
print(f"MS-SSIM : {'enabled' if _HAVE_MSSSIM else 'disabled'}")
print()
if _HAVE_LPIPS and _lpips_fn is not None:
    _lpips_fn = _lpips_fn.to(DEVICE)
# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════
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
class UNet(nn.Module):
    """Universal U-Net -- works for both 3ch and 5ch input. Used by Models A-D."""
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.c1 = ConvBlock(in_ch, 64,   0.1); self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(64,   128,   0.1); self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(128,  256,   0.2); self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(256,  512,   0.2); self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(512,  1024,  0.3)
        self.u6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c6 = ConvBlock(1024, 512,  0.2)
        self.u7 = nn.ConvTranspose2d(512,  256, 2, stride=2)
        self.c7 = ConvBlock(512,  256,  0.2)
        self.u8 = nn.ConvTranspose2d(256,  128, 2, stride=2)
        self.c8 = ConvBlock(256,  128,  0.1)
        self.u9 = nn.ConvTranspose2d(128,   64, 2, stride=2)
        self.c9 = ConvBlock(128,   64,  0.1)
        self.out = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        c1 = self.c1(x);  p1 = self.p1(c1)
        c2 = self.c2(p1); p2 = self.p2(c2)
        c3 = self.c3(p2); p3 = self.p3(c3)
        c4 = self.c4(p3); p4 = self.p4(c4)
        c5 = self.c5(p4)
        c6 = self.c6(torch.cat([self.u6(c5), c4], dim=1))
        c7 = self.c7(torch.cat([self.u7(c6), c3], dim=1))
        c8 = self.c8(torch.cat([self.u8(c7), c2], dim=1))
        c9 = self.c9(torch.cat([self.u9(c8), c1], dim=1))
        return torch.sigmoid(self.out(c9))


# ── Model E's architecture (residual blocks + ASPP + attention gates) ──
class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.proj  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        res = self.proj(x)
        x   = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x   = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.drop(F.relu(x + res, inplace=True))

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
    """Model E -- residual blocks + ASPP bottleneck + attention-gated skips."""
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        self.e1 = ResConvBlock(in_ch, 64,   0.1); self.p1 = nn.MaxPool2d(2)
        self.e2 = ResConvBlock(64,   128,   0.1); self.p2 = nn.MaxPool2d(2)
        self.e3 = ResConvBlock(128,  256,   0.2); self.p3 = nn.MaxPool2d(2)
        self.e4 = ResConvBlock(256,  512,   0.2); self.p4 = nn.MaxPool2d(2)
        self.bottleneck = ASPP(512, 1024)
        self.ag4 = AttentionGate(f_g=1024, f_x=512, f_int=256)
        self.ag3 = AttentionGate(f_g=512,  f_x=256, f_int=128)
        self.ag2 = AttentionGate(f_g=256,  f_x=128, f_int=64)
        self.ag1 = AttentionGate(f_g=128,  f_x=64,  f_int=32)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.d4 = ResConvBlock(1024, 512, 0.2)
        self.u3 = nn.ConvTranspose2d(512,  256, 2, stride=2); self.d3 = ResConvBlock(512,  256, 0.2)
        self.u2 = nn.ConvTranspose2d(256,  128, 2, stride=2); self.d2 = ResConvBlock(256,  128, 0.1)
        self.u1 = nn.ConvTranspose2d(128,   64, 2, stride=2); self.d1 = ResConvBlock(128,   64, 0.1)
        self.out_main = nn.Conv2d(64, out_ch, 1)
        self.out_aux  = nn.Conv2d(256, out_ch, 1)   # training-only, unused at eval
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


# ── Model F's architecture (plain autoencoder, no skip connections) ──
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


# ── Model G's architecture (Context-Encoder GAN -- only the generator is needed at eval) ──
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


# ── Model H's architecture (Partial-Conv U-Net) ──
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


# ── Model I's architecture (ViT Inpainter -- pure transformer) ──
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


def load_model(ckpt_path, in_ch, arch='unet'):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # honour in_ch / arch stored inside checkpoint if present
    in_ch = ckpt.get('in_ch', in_ch)
    arch  = ckpt.get('arch', arch)

    if arch == 'advanced':
        model = AdvancedUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    elif arch == 'plain_ae':
        model = PlainAutoencoder(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    elif arch == 'gan':
        model = CEGenerator(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['generator_state_dict'])   # only G is needed at eval
    elif arch == 'pconv':
        model = PConvUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    elif arch == 'vit':
        model = ViTInpainter(in_ch=in_ch, out_ch=3, img_size=TARGET_SIZE[0]).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model = UNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    print(f"    checkpoint : {ckpt_path}")
    print(f"    epoch      : {ckpt.get('epoch', '?')}  "
          f"best_val : {ckpt.get('best_val', '?')}")
    print(f"    in_ch      : {in_ch}   arch: {arch}")
    return model, in_ch, arch
# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# Supports both 3-channel (Base) and 5-channel (B / C / D / E) input
# ═══════════════════════════════════════════════════════════════════════════════
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
class EvalDataset(Dataset):
    """
    Returns:
        x         -- (in_ch, H, W)  model input
        y         -- (3,     H, W)  clean target
        occ_mask  -- (1,     H, W)  occlusion mask
        bbox_mask -- (1,     H, W)  sign bounding-box mask
    """
    def __init__(self, samples, annot_dir, mask_dir,
                 target_size=(256, 256), in_ch=5):
        self.samples     = samples
        self.annot_dir   = annot_dir
        self.mask_dir    = mask_dir
        self.target_size = target_size
        self.in_ch       = in_ch
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
            H, W = self.target_size[1], self.target_size[0]
            return np.zeros((H, W), dtype=np.float32)
        mask = cv2.resize(mask, self.target_size,
                          interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)
    def __getitem__(self, idx):
        s = self.samples[idx]
        occ_img,   _           = self._load_rgb(s['occ_path'])
        clean_img, original_hw = self._load_rgb(s['clean_path'])
        occ_mask               = self._load_occ_mask(s.get('mask_path', ''))
        Ht, Wt   = self.target_size[1], self.target_size[0]
        base     = os.path.splitext(os.path.basename(s['clean_path']))[0]
        xml_path = os.path.join(self.annot_dir, base + '.xml')
        bboxes   = parse_annotation(xml_path, original_hw, (Ht, Wt))
        bbox_mask = (np.ones((Ht, Wt), dtype=np.float32)
                     if len(bboxes) == 0
                     else make_bbox_mask(Ht, Wt, bboxes))
        occ_img   = np.clip(occ_img,   0, 1)
        occ_mask  = np.clip(occ_mask,  0, 1)
        bbox_mask = np.clip(bbox_mask, 0, 1)
        if self.in_ch == 3:
            x = occ_img.astype(np.float32)
        else:
            x = np.concatenate([
                occ_img,
                occ_mask[..., None],
                bbox_mask[..., None],
            ], axis=-1).astype(np.float32)
        return (
            torch.from_numpy(x.transpose(2, 0, 1)).float(),
            torch.from_numpy(clean_img.transpose(2, 0, 1)).float(),
            torch.from_numpy(occ_mask).unsqueeze(0).float(),
            torch.from_numpy(bbox_mask).unsqueeze(0).float(),
        )
def collate_fn(batch):
    xs, ys, oms, bms = zip(*batch)
    return (torch.stack(xs), torch.stack(ys),
            torch.stack(oms), torch.stack(bms))
# ═══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _gauss_win(size=11, sigma=1.5, C=3, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g.view(1, 1, 1, size).transpose(2, 3) @ g.view(1, 1, 1, size)
    return w.expand(C, 1, size, size).contiguous()
def ssim_map(x, y, ws=11, sigma=1.5, eps=1e-8):
    x = x.clamp(0, 1); y = y.clamp(0, 1)
    C  = x.shape[1]
    win = _gauss_win(ws, sigma, C, device=x.device)
    pad = ws // 2
    mu_x  = F.conv2d(x,   win, padding=pad, groups=C)
    mu_y  = F.conv2d(y,   win, padding=pad, groups=C)
    sg_x2 = (F.conv2d(x*x, win, padding=pad, groups=C) - mu_x**2).clamp(min=0)
    sg_y2 = (F.conv2d(y*y, win, padding=pad, groups=C) - mu_y**2).clamp(min=0)
    sg_xy =  F.conv2d(x*y, win, padding=pad, groups=C) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    num = (2*mu_x*mu_y + c1) * (2*sg_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sg_x2 + sg_y2 + c2) + eps
    return num / den
def batch_psnr(p, g):
    mse = ((p-g)**2).mean(dim=(1,2,3)).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).cpu().numpy()
def batch_ssim(p, g):
    return ssim_map(p, g).mean(dim=(1,2,3)).cpu().numpy()
def _ms_ssim_builtin(x, y, ws=11, sigma=1.5, eps=1e-8):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    x = x.clamp(0, 1).float()
    y = y.clamp(0, 1).float()
    log_ms = torch.zeros(x.shape[0], device=x.device)
    ssim_final = None
    for i, w in enumerate(weights):
        if x.shape[-1] < 11 or x.shape[-2] < 11:
            break
        C   = x.shape[1]
        win = _gauss_win(ws, sigma, C, device=x.device)
        pad = ws // 2
        mu_x  = F.conv2d(x,    win, padding=pad, groups=C)
        mu_y  = F.conv2d(y,    win, padding=pad, groups=C)
        sg_x2 = (F.conv2d(x*x, win, padding=pad, groups=C) - mu_x**2).clamp(min=0)
        sg_y2 = (F.conv2d(y*y, win, padding=pad, groups=C) - mu_y**2).clamp(min=0)
        sg_xy =  F.conv2d(x*y, win, padding=pad, groups=C) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        ssim_map_ = ((2*mu_x*mu_y + c1) * (2*sg_xy + c2)) / \
                   ((mu_x**2 + mu_y**2 + c1) * (sg_x2 + sg_y2 + c2) + eps)
        cs_map   = (2*sg_xy + c2) / (sg_x2 + sg_y2 + c2 + eps)
        ssim_mean = ssim_map_.mean(dim=(1, 2, 3)).clamp(min=1e-8)
        cs_mean   = cs_map.mean(dim=(1, 2, 3)).clamp(min=1e-8)
        ssim_final = ssim_mean
        if i < len(weights) - 1:
            log_ms = log_ms + w * torch.log(cs_mean)
        else:
            log_ms = log_ms + w * torch.log(ssim_mean)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)
    return torch.exp(log_ms).cpu().numpy()
def batch_ms_ssim(p, g):
    if _HAVE_MSSSIM:
        return np.array([
            _ms_ssim_fn(p[i:i+1], g[i:i+1], data_range=1.0,
                        size_average=True).item()
            for i in range(p.shape[0])
        ])
    with torch.no_grad():
        return _ms_ssim_builtin(p.to(DEVICE), g.to(DEVICE))
def batch_lpips(p, g):
    if _HAVE_LPIPS and _lpips_fn is not None:
        with torch.no_grad():
            v = _lpips_fn((p*2-1).to(DEVICE),
                          (g*2-1).to(DEVICE)).squeeze().cpu().numpy()
        return np.atleast_1d(v)
    def sobel(x):
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           dtype=torch.float32, device=x.device)
        ky = kx.t()
        kx = kx.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)
        ky = ky.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)
        return torch.sqrt(F.conv2d(x, kx, padding=1, groups=x.shape[1])**2 +
                          F.conv2d(x, ky, padding=1, groups=x.shape[1])**2 + 1e-8)
    with torch.no_grad():
        v = (sobel(p.to(DEVICE)) - sobel(g.to(DEVICE))).abs() \
              .mean(dim=(1,2,3)).cpu().numpy()
    return np.atleast_1d(v)
def batch_mse(p, g):
    return ((p-g)**2).mean(dim=(1,2,3)).cpu().numpy()
def batch_mae(p, g):
    return (p-g).abs().mean(dim=(1,2,3)).cpu().numpy()
def batch_rmse(p, g):
    return np.sqrt(batch_mse(p, g))
def masked_psnr(p, g, mask):
    m   = mask.expand_as(p)
    n   = m.sum(dim=(1,2,3)).clamp(min=1)
    mse = ((p-g)**2 * m).sum(dim=(1,2,3)) / n
    return (10 * torch.log10(1.0 / mse.clamp(min=1e-10))).cpu().numpy()
def masked_ssim(p, g, mask):
    sm = ssim_map(p, g)
    m  = mask.expand_as(sm)
    n  = m.sum(dim=(1,2,3)).clamp(min=1)
    return ((sm * m).sum(dim=(1,2,3)) / n).cpu().numpy()
def masked_mae(p, g, mask):
    m = mask.expand_as(p)
    n = m.sum(dim=(1,2,3)).clamp(min=1)
    return ((p-g).abs() * m).sum(dim=(1,2,3)).cpu().numpy() / n.cpu().numpy()
# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_test_samples(split_path):
    with open(split_path) as f:
        split = json.load(f)
    samples = split.get("test", split.get("val", []))
    print(f"Test samples loaded: {len(samples)}")
    return samples
def filter_by_level(samples, level):
    filtered = []
    for s in samples:
        if 'occ_level' in s:
            if int(s['occ_level']) == level:
                filtered.append(s)
            continue
        occ_path = s.get('occ_path', '').replace('\\', '/')
        parts    = occ_path.split('/')
        if (str(level) in parts
                or f"_{level}_" in occ_path
                or f"occ{level}" in occ_path):
            filtered.append(s)
    return filtered
# ═══════════════════════════════════════════════════════════════════════════════
# CORE EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_level(model, samples, in_ch, arch='unet'):
    ds = EvalDataset(samples, ANNOT_DIR, MASK_DIR, TARGET_SIZE, in_ch=in_ch)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_fn, drop_last=False)
    # GPU warmup -- prevents first-batch CUDA compile spike
    if DEVICE.type == 'cuda':
        with torch.no_grad():
            if arch == 'pconv':
                _ = model(torch.zeros(1, in_ch, *TARGET_SIZE, device=DEVICE),
                         torch.ones(1, 1, *TARGET_SIZE, device=DEVICE))
            else:
                _ = model(torch.zeros(1, in_ch, *TARGET_SIZE, device=DEVICE))
        torch.cuda.synchronize()
    acc = {k: [] for k in ['psnr','ssim','ms_ssim','lpips',
                             'mse','mae','rmse',
                             'psnr_mask','ssim_mask','mae_mask']}
    total_time = 0.0
    with torch.no_grad():
        for x, y, occ_mask, _ in loader:
            x        = x.to(DEVICE)
            y        = y.to(DEVICE)
            occ_mask = occ_mask.to(DEVICE)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0   = time.perf_counter()
            # NOTE: AdvancedUNet(x) with default return_aux=False, and every
            # other single-input model (UNet/PlainAutoencoder/CEGenerator/
            # ViTInpainter) all share the plain model(x) call signature.
            # Only PConvUNet needs a second (valid-mask) argument.
            if arch == 'pconv':
                valid_mask = 1.0 - occ_mask
                pred = model(x, valid_mask).clamp(0, 1)
            else:
                pred = model(x).clamp(0, 1)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            acc['psnr'     ].extend(batch_psnr   (pred, y).tolist())
            acc['ssim'     ].extend(batch_ssim   (pred, y).tolist())
            acc['ms_ssim'  ].extend(batch_ms_ssim(pred, y).tolist())
            acc['lpips'    ].extend(batch_lpips  (pred, y).tolist())
            acc['mse'      ].extend(batch_mse    (pred, y).tolist())
            acc['mae'      ].extend(batch_mae    (pred, y).tolist())
            acc['rmse'     ].extend(batch_rmse   (pred, y).tolist())
            acc['psnr_mask'].extend(masked_psnr  (pred, y, occ_mask).tolist())
            acc['ssim_mask'].extend(masked_ssim  (pred, y, occ_mask).tolist())
            acc['mae_mask' ].extend(masked_mae   (pred, y, occ_mask).tolist())
    n      = len(acc['psnr'])
    means  = {k: float(np.mean(v)) for k, v in acc.items()}
    means['_time']   = total_time
    means['_avg_ms'] = (total_time / n * 1000) if n > 0 else 0.0
    return means, total_time, n
# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL WRITER -- one file per model, identical to enhanced_model.xlsx structure
# ═══════════════════════════════════════════════════════════════════════════════
def _fill(hex_col):
    return PatternFill("solid", fgColor=hex_col)
def _border():
    s = Side(style='thin', color='BBBBBB')
    return Border(left=s, right=s, top=s, bottom=s)
def _center():
    return Alignment(horizontal='center', vertical='center', wrap_text=True)
_HEADERS = [
    'Level (%)', 'N images',
    'PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS',
    'MSE', 'MAE', 'RMSE',
    'PSNR_mask', 'SSIM_mask', 'MAE_mask',
    'Total_sec', 'Avg_ms/img',
]
_GROUP_SPANS = [
    ("Full Image -- Quality", "7B68EE", 3, 6),
    ("Full Image -- Error",   "E8735A", 7, 9),
    ("Occluded Region Only",  "3CB371", 10, 12),
    ("Inference Timing",      "4682B4", 13, 14),
]
_DATA_FILLS = (
    ["FFFFFF", "FFFFFF"]
    + ["EAE6F7"] * 4
    + ["FAE0D8"] * 3
    + ["D4EFE0"] * 3
    + ["D4E8F5"] * 2
)
_BOLD_COLS = {3, 4, 10, 11}
def _write_model_sheet(ws, model_name, results_by_level):
    ws.merge_cells("A1:N1")
    ws["A1"].value     = f"{model_name} -- Test Split Evaluation"
    ws["A1"].font      = Font(bold=True, color="FFFFFF", size=13)
    ws["A1"].fill      = _fill("002060")
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 22
    ws["A3"].value = ""; ws["B3"].value = ""
    for label, color, c1, c2 in _GROUP_SPANS:
        cl1 = get_column_letter(c1); cl2 = get_column_letter(c2)
        ws.merge_cells(f"{cl1}3:{cl2}3")
        cell            = ws[f"{cl1}3"]
        cell.value      = label
        cell.fill       = _fill(color)
        cell.font       = Font(bold=True, color="FFFFFF", size=10)
        cell.alignment  = _center()
        cell.border     = _border()
    ws.row_dimensions[3].height = 16
    for col_i, h in enumerate(_HEADERS, 1):
        cell            = ws.cell(row=4, column=col_i, value=h)
        cell.font       = Font(bold=True, size=9)
        cell.fill       = _fill("D9D9D9")
        cell.alignment  = _center()
        cell.border     = _border()
    for row_i, level in enumerate(OCC_LEVELS, 5):
        r  = results_by_level[level]
        m  = r['metrics']
        n  = r['n']
        row_vals = [
            level, n,
            round(m['psnr'],      6), round(m['ssim'],      6),
            round(m['ms_ssim'],   6), round(m['lpips'],     6),
            round(m['mse'],       6), round(m['mae'],       6),
            round(m['rmse'],      6),
            round(m['psnr_mask'], 6), round(m['ssim_mask'], 6),
            round(m['mae_mask'],  6),
            round(m['_time'],     4), round(m['_avg_ms'],   4),
        ]
        for col_i, val in enumerate(row_vals, 1):
            cell            = ws.cell(row=row_i, column=col_i, value=val)
            cell.alignment  = _center()
            cell.border     = _border()
            cell.fill       = _fill(_DATA_FILLS[col_i-1])
            if col_i in _BOLD_COLS:
                cell.font   = Font(bold=True, color="002060", size=9)
            else:
                cell.font   = Font(size=9)
    for i, w in enumerate([10,9,11,9,9,9,9,9,9,11,11,10,11,12], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
def write_model_excel(model_name, results_by_level, out_path):
    if not _HAVE_XLSX:
        print("    [SKIP] openpyxl not available")
        return
    wb = openpyxl.Workbook()
    ws1 = wb.active; ws1.title = "Metrics Summary"
    _write_model_sheet(ws1, model_name, results_by_level)
    ws2 = wb.create_sheet("Per-Metric Detail")
    ws2.append(["Occlusion Level", "Metric", "Value", "Direction", "Category"])
    for cell in ws2[1]:
        cell.font = Font(bold=True); cell.fill = _fill("D9D9D9")
        cell.alignment = _center(); cell.border = _border()
    _detail_rows = [
        ('PSNR (dB)',   'psnr',      'higher better', 'Full image'),
        ('SSIM',        'ssim',      'higher better', 'Full image'),
        ('MS-SSIM',     'ms_ssim',   'higher better', 'Full image'),
        ('LPIPS',       'lpips',     'lower better',  'Full image'),
        ('MSE',         'mse',       'lower better',  'Full image'),
        ('MAE',         'mae',       'lower better',  'Full image'),
        ('RMSE',        'rmse',      'lower better',  'Full image'),
        ('PSNR masked', 'psnr_mask', 'higher better', 'Occluded region'),
        ('SSIM masked', 'ssim_mask', 'higher better', 'Occluded region'),
        ('MAE masked',  'mae_mask',  'lower better',  'Occluded region'),
        ('Total_sec',   '_time',     'lower better',  'Timing'),
        ('Avg ms/img',  '_avg_ms',   'lower better',  'Timing'),
    ]
    for level in OCC_LEVELS:
        m = results_by_level[level]['metrics']
        for label, key, direction, cat in _detail_rows:
            ws2.append([f"{level}%", label, round(m[key], 6), direction, cat])
    for col in ws2.columns:
        ws2.column_dimensions[col[0].column_letter].width = 18
    ws3 = wb.create_sheet("Charts")
    ws3.append(["Level","PSNR","SSIM","MS-SSIM","LPIPS",
                 "MSE","MAE","RMSE","PSNR_mask","SSIM_mask","MAE_mask",
                 "Total_sec","Avg_ms_img"])
    for cell in ws3[1]:
        cell.font = Font(bold=True); cell.fill = _fill("D9D9D9")
    for level in OCC_LEVELS:
        m = results_by_level[level]['metrics']
        ws3.append([f"{level}%",
                    round(m['psnr'],6),     round(m['ssim'],6),
                    round(m['ms_ssim'],6),  round(m['lpips'],6),
                    round(m['mse'],6),      round(m['mae'],6),
                    round(m['rmse'],6),     round(m['psnr_mask'],6),
                    round(m['ssim_mask'],6),round(m['mae_mask'],6),
                    round(m['_time'],4),    round(m['_avg_ms'],4)])
    ws4 = wb.create_sheet("Statistics")
    ws4.append(["Summary statistics -- test split", None, None, None])
    ws4.append(["Metric", "Mean", "Min", "Max"])
    for cell in ws4[2]:
        cell.font = Font(bold=True); cell.fill = _fill("D9D9D9")
    _stat_keys = [
        ('PSNR (dB) (higher better)',  'psnr'),
        ('SSIM (higher better)',        'ssim'),
        ('MS-SSIM (higher better)',     'ms_ssim'),
        ('LPIPS (lower better)',        'lpips'),
        ('MSE (lower better)',          'mse'),
        ('MAE (lower better)',          'mae'),
        ('RMSE (lower better)',         'rmse'),
        ('PSNR masked (higher better)', 'psnr_mask'),
        ('SSIM masked (higher better)', 'ssim_mask'),
        ('MAE masked (lower better)',   'mae_mask'),
        ('Total inference sec',         '_time'),
        ('Avg ms per image',            '_avg_ms'),
    ]
    for label, key in _stat_keys:
        vals = [results_by_level[lv]['metrics'][key] for lv in OCC_LEVELS]
        ws4.append([label,
                    round(float(np.mean(vals)), 4),
                    round(float(np.min(vals)),  4),
                    round(float(np.max(vals)),  4)])
    for col in ws4.columns:
        ws4.column_dimensions[col[0].column_letter].width = 30
    wb.save(out_path)
    print(f"    Excel saved  -> {out_path}")
# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON EXCEL -- all 5 models side by side
# ═══════════════════════════════════════════════════════════════════════════════
_MODEL_COLORS = ["7B68EE", "E8735A", "3CB371", "4682B4", "97C459",
                 "B4B2A9", "F0997B", "5DCAA5", "EF9F27"]
def write_comparison_excel(all_results, out_path):
    if not _HAVE_XLSX:
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "All Models Comparison"
    model_names = list(all_results.keys())
    col_labels  = ['PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS',
                   'MSE', 'MAE', 'RMSE',
                   'PSNR_mask', 'SSIM_mask', 'MAE_mask',
                   'Total_sec', 'Avg_ms/img']
    col_keys    = ['psnr','ssim','ms_ssim','lpips',
                   'mse','mae','rmse',
                   'psnr_mask','ssim_mask','mae_mask',
                   '_time','_avg_ms']
    n_mc = len(col_labels)
    ws.cell(1, 1, "Level").font      = Font(bold=True, size=10)
    ws.cell(1, 1).alignment          = _center()
    ws.cell(1, 1).fill               = _fill("D9D9D9")
    ws.cell(1, 1).border             = _border()
    col = 2
    for mi, name in enumerate(model_names):
        color   = _MODEL_COLORS[mi % len(_MODEL_COLORS)]
        c_start = col
        for j in range(n_mc):
            ws.cell(1, col).fill      = _fill(color)
            ws.cell(1, col).font      = Font(bold=True, color='FFFFFF', size=9)
            ws.cell(1, col).alignment = _center()
            ws.cell(1, col).border    = _border()
            col += 1
        ws.merge_cells(start_row=1, start_column=c_start,
                       end_row=1,   end_column=col - 1)
        ws.cell(1, c_start).value     = name
        ws.cell(1, c_start).alignment = _center()
    ws.cell(2, 1, "").fill = _fill("D9D9D9")
    col = 2
    for mi in range(len(model_names)):
        for lbl in col_labels:
            cell            = ws.cell(2, col, lbl)
            cell.font       = Font(bold=True, size=8)
            cell.fill       = _fill("EEEEEE")
            cell.alignment  = _center()
            cell.border     = _border()
            col += 1
    for row_i, level in enumerate(OCC_LEVELS, 3):
        ws.cell(row_i, 1, f"{level}%").alignment = _center()
        ws.cell(row_i, 1).border                 = _border()
        ws.cell(row_i, 1).font                   = Font(bold=True, size=9)
        col = 2
        for mi, name in enumerate(model_names):
            m     = all_results[name][level]['metrics']
            light = ["F0EEFF", "FEF0EC", "E8F8EE", "E8F4FB", "F1F8E5",
                    "F5F4F2", "FDEEE9", "E5F7F1", "FDF3E3"][mi % 9]
            for key in col_keys:
                cell            = ws.cell(row_i, col, round(m.get(key, 0), 6))
                cell.alignment  = _center()
                cell.border     = _border()
                cell.fill       = _fill(light)
                cell.font       = Font(size=9)
                col += 1
    ws2 = wb.create_sheet("Average Across Levels")
    ws2.append(["Model"] + col_labels)
    for cell in ws2[1]:
        cell.font = Font(bold=True); cell.fill = _fill("D9D9D9")
        cell.alignment = _center(); cell.border = _border()
    for mi, name in enumerate(model_names):
        row = [name]
        for key in col_keys:
            vals = [all_results[name][lv]['metrics'].get(key, 0)
                    for lv in OCC_LEVELS]
            row.append(round(float(np.mean(vals)), 6))
        ws2.append(row)
        for cell in ws2[ws2.max_row]:
            cell.alignment = _center(); cell.border = _border()
            cell.fill      = _fill(_MODEL_COLORS[mi % len(_MODEL_COLORS)])
            cell.font      = Font(color="FFFFFF", size=9)
    for col in ws2.columns:
        ws2.column_dimensions[col[0].column_letter].width = 16
    n_cols_ws = 1 + len(model_names) * n_mc
    for i in range(1, n_cols_ws + 1):
        ws.column_dimensions[get_column_letter(i)].width = 13
    wb.save(out_path)
    print(f"    Comparison Excel saved -> {out_path}")
# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading test split...")
    all_test_samples = load_test_samples(SPLIT_PATH)
    print("\nModel availability check:")
    runnable = []
    for model_name, ckpt_path, in_ch, arch in MODELS:
        exists = os.path.exists(ckpt_path)
        status = "[FOUND]" if exists else "[MISSING -- will skip]"
        print(f"  {status:<25} {model_name}")
        print(f"  {'':25} {ckpt_path}")
        if exists:
            runnable.append((model_name, ckpt_path, in_ch, arch))
    print(f"\nWill evaluate {len(runnable)} / {len(MODELS)} models.\n")
    if not runnable:
        print("No checkpoints found. Exiting.")
        return
    all_results = {}
    for model_name, ckpt_path, cfg_in_ch, cfg_arch in runnable:
        print(f"\n{'='*62}")
        print(f"  {model_name}")
        print(f"{'='*62}")
        try:
            model, in_ch, resolved_arch = load_model(ckpt_path, cfg_in_ch, cfg_arch)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            continue
        results_by_level = {}
        for level in OCC_LEVELS:
            samples = filter_by_level(all_test_samples, level)
            if len(samples) == 0:
                print(f"  [WARN] Level {level}%: no tagged samples -- "
                      f"using all {len(all_test_samples)} test samples")
                samples = all_test_samples
            print(f"  Level {level:3d}% | {len(samples):4d} imgs ...",
                  end="", flush=True)
            metrics, total_time, n = evaluate_level(model, samples, in_ch, arch=resolved_arch)
            results_by_level[level] = {
                'metrics': metrics,
                'time':    total_time,
                'n':       n,
            }
            avg_ms = (total_time / n * 1000) if n > 0 else 0.0
            print(f"  PSNR={metrics['psnr']:.4f}"
                  f"  SSIM={metrics['ssim']:.4f}"
                  f"  MS-SSIM={metrics['ms_ssim']:.4f}"
                  f"  LPIPS={metrics['lpips']:.4f}"
                  f"  Total={total_time:.2f}s"
                  f"  Avg={avg_ms:.2f}ms/img")
        all_results[model_name] = results_by_level
        safe  = (model_name.replace(" ", "_")
                            .replace("--", "")
                            .replace("/", "")
                            .replace("__", "_"))
        xlsx_path = os.path.join(OUT_DIR, f"{safe}.xlsx")
        write_model_excel(model_name, results_by_level, xlsx_path)
        json_path = os.path.join(OUT_DIR, f"{safe}.json")
        with open(json_path, 'w') as f:
            json.dump(results_by_level, f, indent=2)
        print(f"    JSON saved   -> {json_path}")
    if all_results:
        print(f"\n{'='*62}")
        print("  Writing combined comparison Excel...")
        comp_path = os.path.join(OUT_DIR, "comparison_all_models.xlsx")
        write_comparison_excel(all_results, comp_path)
        master_path = os.path.join(OUT_DIR, "all_results.json")
        with open(master_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"    Master JSON  -> {master_path}")
    print(f"\n{'='*62}")
    print("  FINAL SUMMARY")
    print(f"{'='*62}")
    print(f"{'Model':<32} {'Lvl':>5} {'PSNR':>8} {'SSIM':>7}"
          f" {'MS-SSIM':>8} {'LPIPS':>7} {'Avg ms':>8}")
    print("-" * 78)
    for mname, res in all_results.items():
        for level in OCC_LEVELS:
            m      = res[level]['metrics']
            avg_ms = m['_avg_ms']
            print(f"{mname:<32} {level:>4}%"
                  f"  {m['psnr']:>8.4f}"
                  f"  {m['ssim']:>7.4f}"
                  f"  {m['ms_ssim']:>8.4f}"
                  f"  {m['lpips']:>7.4f}"
                  f"  {avg_ms:>7.2f}ms")
        print()
    print(f"\nAll files saved to:  {OUT_DIR}/")
if __name__ == "__main__":
    main()
