"""
evaluate_baseline_models.py
============================
Evaluates Models F-I (the non-U-Net baselines from train_baseline_models.py)
on the test split, producing one .xlsx per model with a "Metrics Summary"
sheet broken down by occlusion level -- the SAME layout used for Models A-E
(Level, N, PSNR, SSIM, MSSSIM, LPIPS, MSE, MAE, RMSE, PSNR_mask, SSIM_mask,
MAE_mask, Total_sec, Avg_ms) -- so all 9 models can be compared directly.

Requires (in addition to train_baseline_models.py's dependencies):
    pip install scikit-image pytorch-msssim lpips openpyxl pandas

IMPORTANT -- occlusion level detection:
This script needs to know which occlusion level (0/10/.../80%) each test
sample belongs to. Your data_split.json samples may or may not already
carry this. `get_level()` below tries, in order:
    1. an explicit `level` key on the sample dict (e.g. {"level": 40})
    2. a numeric folder segment in occ_path (.../occluded/40/img.png)
    3. a "_<n>pct" / "lvl<n>" / "level_<n>" pattern in the filename
If none of these match your naming convention, edit `get_level()` --
it is the ONLY function you should need to change to fit your dataset.

Usage:
    python evaluate_baseline_models.py
"""

import os, re, json, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from pytorch_msssim import ssim as t_ssim, ms_ssim as t_ms_ssim

import lpips as lpips_lib

from train_baseline_models import (
    SignDataset, PlainAutoencoder, CEGenerator, PConvUNet, ViTInpainter,
    SPLIT_PATH, TARGET_SIZE, DEVICE, load_split,
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

OCCLUSION_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80]

MODELS_TO_EVAL = [
    {
        "name"      : "Model_F_Plain_Autoencoder",
        "arch"      : "plain_ae",
        "ckpt_path" : "model_F_plain_autoencoder/best_model_F.pth",
        "xlsx_path" : "Model_F_Plain_Autoencoder.xlsx",
    },
    {
        "name"      : "Model_G_ContextEncoder_GAN",
        "arch"      : "gan",
        "ckpt_path" : "model_G_context_encoder_gan/best_model_G.pth",
        "xlsx_path" : "Model_G_ContextEncoder_GAN.xlsx",
    },
    {
        "name"      : "Model_H_PartialConv_UNet",
        "arch"      : "pconv",
        "ckpt_path" : "model_H_partial_conv_unet/best_model_H.pth",
        "xlsx_path" : "Model_H_PartialConv_UNet.xlsx",
    },
    {
        "name"      : "Model_I_ViT_Inpainter",
        "arch"      : "vit",
        "ckpt_path" : "model_I_vit_inpainter/best_model_I.pth",
        "xlsx_path" : "Model_I_ViT_Inpainter.xlsx",
    },
]

print(f"Device : {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# OCCLUSION-LEVEL DETECTION  -- EDIT THIS TO MATCH YOUR DATASET
# ═══════════════════════════════════════════════════════════════

_LEVEL_PATTERNS = [
    re.compile(r'[/\\](\d{1,3})[/\\]'),                 # .../occluded/40/img.png
    re.compile(r'lvl[_\-]?(\d{1,3})', re.IGNORECASE),   # img_lvl40.png
    re.compile(r'level[_\-]?(\d{1,3})', re.IGNORECASE), # img_level_40.png
    re.compile(r'(\d{1,3})pct', re.IGNORECASE),         # img_40pct.png
    re.compile(r'_(\d{1,3})_'),                         # img_40_001.png
]

def get_level(sample):
    """Return this sample's occlusion level (int, e.g. 40) or None if unknown."""
    if 'level' in sample:
        return int(sample['level'])
    path = sample.get('occ_path', '')
    for pat in _LEVEL_PATTERNS:
        m = pat.search(path)
        if m:
            val = int(m.group(1))
            if val in OCCLUSION_LEVELS:
                return val
    return None


def group_by_level(samples):
    groups = {lvl: [] for lvl in OCCLUSION_LEVELS}
    unmatched = 0
    for s in samples:
        lvl = get_level(s)
        if lvl is None:
            unmatched += 1
            continue
        groups.setdefault(lvl, []).append(s)
    if unmatched:
        print(f"  WARNING: {unmatched} test samples had no detectable occlusion "
              f"level and were skipped -- check get_level() in this script.")
    return groups


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(cfg):
    arch = cfg['arch']
    ckpt = torch.load(cfg['ckpt_path'], map_location=DEVICE)
    in_ch = ckpt.get('in_ch', 5)

    if arch == 'plain_ae':
        model = PlainAutoencoder(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    elif arch == 'gan':
        model = CEGenerator(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['generator_state_dict'])
    elif arch == 'pconv':
        model = PConvUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    elif arch == 'vit':
        model = ViTInpainter(in_ch=in_ch, out_ch=3, img_size=TARGET_SIZE[0]).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        raise ValueError(f"Unknown arch {arch}")

    model.eval()
    return model, arch


@torch.no_grad()
def run_inference(model, arch, x, occ_mask):
    if arch == 'pconv':
        valid_mask = 1.0 - occ_mask
        return model(x, valid_mask)
    return model(x)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

_lpips_fn = None
def get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE)
        _lpips_fn.eval()
    return _lpips_fn


def compute_metrics(pred, gt, occ_mask):
    """
    pred, gt   : (1,3,H,W) tensors in [0,1]
    occ_mask   : (1,1,H,W) tensor, 1 = occluded/hidden region
    Returns a dict of scalar metrics for this single image.
    """
    pred_c = pred.clamp(0, 1)
    gt_c   = gt.clamp(0, 1)

    pred_np = pred_c[0].permute(1, 2, 0).cpu().numpy()
    gt_np   = gt_c[0].permute(1, 2, 0).cpu().numpy()
    mask_np = occ_mask[0, 0].cpu().numpy() > 0.5

    psnr_full = sk_psnr(gt_np, pred_np, data_range=1.0)
    ssim_full = sk_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2)

    with torch.no_grad():
        ms_ssim_full = t_ms_ssim(pred_c, gt_c, data_range=1.0, size_average=True).item()

    mse = float(np.mean((pred_np - gt_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - gt_np)))
    rmse = float(np.sqrt(mse))

    if mask_np.sum() > 0:
        pred_m = pred_np[mask_np]
        gt_m   = gt_np[mask_np]
        psnr_mask = sk_psnr(gt_m, pred_m, data_range=1.0)
        mae_mask  = float(np.mean(np.abs(pred_m - gt_m)))
        # SSIM needs a 2D/3D spatial map; compute full-image SSIM map, then
        # average only over the masked pixels for a "masked SSIM".
        _, ssim_map = sk_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2, full=True)
        ssim_mask = float(ssim_map[mask_np].mean())
    else:
        psnr_mask = float('nan')
        ssim_mask = float('nan')
        mae_mask  = float('nan')

    with torch.no_grad():
        lpips_val = get_lpips_fn()(pred_c * 2 - 1, gt_c * 2 - 1).item()

    return {
        'PSNR': psnr_full, 'SSIM': ssim_full, 'MSSSIM': ms_ssim_full, 'LPIPS': lpips_val,
        'MSE': mse, 'MAE': mae, 'RMSE': rmse,
        'PSNR_mask': psnr_mask, 'SSIM_mask': ssim_mask, 'MAE_mask': mae_mask,
    }


# ═══════════════════════════════════════════════════════════════
# EVALUATE ONE MODEL ACROSS ALL LEVELS
# ═══════════════════════════════════════════════════════════════

def evaluate_model(cfg, level_groups):
    print(f"\n{'='*64}")
    print(f"  Evaluating : {cfg['name']}")
    print(f"{'='*64}")

    model, arch = load_model(cfg)
    rows = []

    for lvl in OCCLUSION_LEVELS:
        samples = level_groups.get(lvl, [])
        if not samples:
            print(f"  Level {lvl:>3}% : no test samples found, skipping")
            continue

        ds = SignDataset(samples, TARGET_SIZE, in_ch=5, augment=False)
        per_image_metrics = []
        t0 = time.time()

        for i in range(len(ds)):
            x, y, occ_mask = ds[i]
            x = x.unsqueeze(0).to(DEVICE)
            y = y.unsqueeze(0).to(DEVICE)
            occ_mask = occ_mask.unsqueeze(0).to(DEVICE)

            pred = run_inference(model, arch, x, occ_mask)
            per_image_metrics.append(compute_metrics(pred, y, occ_mask))

        total_sec = time.time() - t0
        n = len(per_image_metrics)
        avg_ms = (total_sec / n) * 1000 if n else float('nan')

        df = pd.DataFrame(per_image_metrics)
        row = {'Level': f"{lvl}%", 'N': n}
        for col in ['PSNR', 'SSIM', 'MSSSIM', 'LPIPS', 'MSE', 'MAE', 'RMSE',
                    'PSNR_mask', 'SSIM_mask', 'MAE_mask']:
            row[col] = df[col].mean()
        row['Total_sec'] = total_sec
        row['Avg_ms'] = avg_ms
        rows.append(row)

        print(f"  Level {lvl:>3}% (n={n:>3})  PSNR={row['PSNR']:.2f}dB  "
              f"SSIM={row['SSIM']:.4f}  MSSSIM={row['MSSSIM']:.4f}  "
              f"LPIPS={row['LPIPS']:.4f}  PSNR_mask={row['PSNR_mask']:.2f}dB  "
              f"avg={avg_ms:.1f}ms")

    summary_df = pd.DataFrame(rows)
    return summary_df


def save_summary_xlsx(summary_df, cfg, path):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # 3 blank/title rows above the table, matching the A-E workbooks'
        # layout (evaluate_all_models.py wrote a small header block there).
        title_df = pd.DataFrame({'A': [f"Metrics Summary — {cfg['name']}", '', '']})
        title_df.to_excel(writer, sheet_name='Metrics Summary', header=False,
                           index=False, startrow=0)
        summary_df.to_excel(writer, sheet_name='Metrics Summary', index=False, startrow=3)
    print(f"  Saved -> {path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading split...")
    _, test_samples = load_split(SPLIT_PATH)
    level_groups = group_by_level(test_samples)

    for lvl, samples in level_groups.items():
        print(f"  Level {lvl:>3}% : {len(samples)} test samples")

    all_summaries = {}
    for cfg in MODELS_TO_EVAL:
        if not os.path.exists(cfg['ckpt_path']):
            print(f"\n  Skipping {cfg['name']} -- checkpoint not found at "
                  f"{cfg['ckpt_path']} (train it first with train_baseline_models.py)")
            continue
        summary_df = evaluate_model(cfg, level_groups)
        save_summary_xlsx(summary_df, cfg, cfg['xlsx_path'])
        all_summaries[cfg['name']] = summary_df

    # ── combined cross-model overview (mean across all levels) ─────
    if all_summaries:
        overview_rows = []
        for name, df in all_summaries.items():
            row = {'Model': name}
            for col in ['PSNR', 'SSIM', 'MSSSIM', 'LPIPS', 'PSNR_mask', 'SSIM_mask', 'Avg_ms']:
                row[col] = df[col].mean()
            overview_rows.append(row)
        overview = pd.DataFrame(overview_rows)
        overview.to_csv('baseline_models_overview.csv', index=False)
        print(f"\n{'='*64}")
        print("  BASELINE MODELS -- OVERVIEW (mean across all occlusion levels)")
        print(f"{'='*64}")
        print(overview.to_string(index=False))
        print("\n  Saved -> baseline_models_overview.csv")
        print("\nCompare this against Models A-E's xlsx files (or their own")
        print("overview) to confirm Model E outperforms every architecture family.")


if __name__ == "__main__":
    main()
