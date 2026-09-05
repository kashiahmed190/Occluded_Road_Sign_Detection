"""
diagnose_G_I.py
=================
Quick diagnostic for Model G and Model I: loads each, runs inference on
a handful of real test images at a chosen occlusion level, and:
  1. Saves the actual inpainted output as a PNG so you can SEE it
  2. Prints tensor stats (min/max/mean/std, NaN/Inf check)
  3. Prints what the detector receives (after the native-resolution
     upsample) so we can tell whether the bug is in the model itself,
     or in how evaluate_detection_fullimage.py processes its output.

Usage:
    python diagnose_G_I.py
"""

import os, json
import cv2
import numpy as np
import torch

import evaluate_all_models as ev
import evaluate_detection_fullimage as ed

DEVICE = ed.DEVICE
LEVEL_TO_CHECK = 40
N_SAMPLES = 3
OUT_DIR = "diagnose_output"
os.makedirs(OUT_DIR, exist_ok=True)


def diagnose_model(name, ckpt_path, arch, in_ch, samples, yolo_detector=None):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    cfg = {"ckpt": ckpt_path, "arch": arch, "in_ch": in_ch}
    model, resolved_arch, resolved_in_ch = ed.load_inpaint_model(cfg, ev)
    print(f"  Loaded. Resolved arch={resolved_arch}  in_ch={resolved_in_ch}")

    for i, s in enumerate(samples[:N_SAMPLES]):
        native = ed.load_rgb_native(s['clean_path'])
        native_hw = native.shape[:2]

        x, occ_mask = ed.build_5ch_input(s)
        x, occ_mask = x.to(DEVICE), occ_mask.to(DEVICE)

        with torch.no_grad():
            pred256 = ed.run_inpaint(model, resolved_arch, resolved_in_ch, x, occ_mask)

        p = pred256.detach().cpu()
        print(f"\n  Sample {i}: {s['clean_path']}")
        print(f"    pred256 shape={tuple(p.shape)}  "
              f"min={p.min().item():.4f}  max={p.max().item():.4f}  "
              f"mean={p.mean().item():.4f}  std={p.std().item():.4f}")
        print(f"    has NaN: {torch.isnan(p).any().item()}   has Inf: {torch.isinf(p).any().item()}")

        pred_native = ed.tensor_to_native_uint8(pred256, native_hw)
        print(f"    pred_native (uint8, what the detector sees) shape={pred_native.shape}  "
              f"min={pred_native.min()}  max={pred_native.max()}  mean={pred_native.mean():.2f}")

        base = os.path.splitext(os.path.basename(s['clean_path']))[0]
        p256_uint8 = (p[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_{base}_256.png"),
                   cv2.cvtColor(p256_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_{base}_native.png"),
                   cv2.cvtColor(pred_native, cv2.COLOR_RGB2BGR))

        occ_uint8 = (ed.load_rgb_256(s['occ_path']) * 255).astype(np.uint8)
        clean_uint8 = (ed.load_rgb_256(s['clean_path']) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_occluded.png"),
                   cv2.cvtColor(occ_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_clean.png"),
                   cv2.cvtColor(clean_uint8, cv2.COLOR_RGB2BGR))

        if yolo_detector is not None:
            dets = yolo_detector.detect(pred_native)
            print(f"    YOLOv5 detections on this output: {dets}")


def main():
    test_samples = ed.load_split(ed.SPLIT_PATH)
    level_groups = ed.group_by_level(test_samples)
    samples = level_groups.get(LEVEL_TO_CHECK, [])
    print(f"Using {len(samples)} samples at level {LEVEL_TO_CHECK}%")
    if not samples:
        print("No samples found at that level -- edit LEVEL_TO_CHECK.")
        return

    yolo_detector = None
    try:
        yolo_detector = ed.YoloDetectorFull(ed.YOLOV5_WEIGHTS, ed.YOLOV5_CONF_THRES, ed.YOLOV5_IOU_THRES)
    except Exception as e:
        print(f"Could not load YOLOv5 detector ({e}) -- skipping detection step, "
              f"will still save images + print tensor stats.")

    registry, _ = ed.build_inpaint_registry()
    reg_by_name = {r['name'].split(' (')[0]: r for r in registry}

    for key in ['Model G', 'Model I']:
        cfg = reg_by_name[key]
        diagnose_model(cfg['name'], cfg['ckpt'], cfg['arch'], cfg['in_ch'], samples, yolo_detector)

    print(f"\n\nSaved images to {OUT_DIR}/ -- open them and look:")
    print("  <name>_<sample>_256.png     -- raw 256x256 model output")
    print("  <name>_<sample>_native.png  -- upsampled to native res (what the detector sees)")
    print("  <sample>_occluded.png       -- the occluded input")
    print("  <sample>_clean.png          -- the ground truth")


if __name__ == "__main__":
    main()
