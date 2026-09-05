"""
evaluate_detection.py
======================
Runs YOLOv5 + Faster R-CNN detection on every model's inpainted output
(Models A-I) across all 9 occlusion levels, and compares against
detection on the raw occluded image (pre-inpaint baseline). This is
the "Detection-recall results" step from the end-to-end pipeline
diagram: it answers "does inpainting actually help the sign get
detected, and which model helps the most?"

Output: one row per (model, level) with:
    N, Recall_YOLO_pre, Recall_YOLO_post, Recall_FRCNN_pre, Recall_FRCNN_post,
    Gain_YOLO (post - pre), Gain_FRCNN (post - pre)
saved to detection_results.xlsx (one sheet per detector) and
detection_results_summary.csv (mean across all levels, per model).

────────────────────────────────────────────────────────────────
BEFORE RUNNING -- 3 things you MUST configure for your setup:
────────────────────────────────────────────────────────────────
1. YOLOV5_REPO_PATH / YOLOV5_WEIGHTS  -- your trained YOLOv5 sign detector
2. FRCNN_WEIGHTS / FRCNN_NUM_CLASSES  -- your trained Faster R-CNN sign detector
3. get_level()                        -- how occlusion level is encoded in
                                          your file paths (see comment below;
                                          same function as in
                                          evaluate_baseline_models.py)

Everything else (model loading for A-I, inpainting, detection-success
logic) should work as-is once those 3 are filled in.
────────────────────────────────────────────────────────────────

Usage:
    python evaluate_detection.py
"""

import os, re, sys, json, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# CONFIG -- EDIT THESE
# ═══════════════════════════════════════════════════════════════

SPLIT_PATH  = os.path.join("traintestsplit", "data_split.json")
TARGET_SIZE = (256, 256)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OCCLUSION_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80]

# Must match prepare_detector_dataset.py's CLASSES order exactly.
CLASSES = ["crosswalk", "speedlimit", "stop", "trafficlight"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def class_id_for(sample):
    """0-indexed class id for this sample's category, or None if unknown
    (falls back to any-box-present detection for that sample)."""
    return CLASS_TO_ID.get(sample.get('category'))


class ConfusionAccumulator:
    """
    Proper per-class precision/recall, not just a detected/missed hit-rate.

    Each image has exactly one true object (one sign). For that image:
      - TP  : the true class appears among the predicted classes
      - FN  : the true class does NOT appear among the predicted classes
      - FP  : every OTHER predicted box whose class != true class (spurious
              detections and misclassifications both count here)

    This also builds a full true-class x predicted-class confusion matrix
    (with an explicit "none" column for images where nothing was predicted
    at all), so you can see exactly what gets confused with what.
    """
    def __init__(self, classes):
        self.classes = classes
        n = len(classes)
        self.tp = np.zeros(n, dtype=int)
        self.fp = np.zeros(n, dtype=int)
        self.fn = np.zeros(n, dtype=int)
        # rows = true class, cols = predicted class (+1 extra col for "none")
        self.matrix = np.zeros((n, n + 1), dtype=int)

    def update(self, true_class_id, predicted_class_ids):
        if true_class_id is None:
            return
        preds = predicted_class_ids
        if true_class_id in preds:
            self.tp[true_class_id] += 1
            self.matrix[true_class_id, true_class_id] += 1
            for p in preds:
                if p != true_class_id and 0 <= p < len(self.classes):
                    self.fp[p] += 1
        else:
            self.fn[true_class_id] += 1
            if preds:
                for p in preds:
                    if 0 <= p < len(self.classes):
                        self.matrix[true_class_id, p] += 1
                        self.fp[p] += 1
            else:
                self.matrix[true_class_id, len(self.classes)] += 1  # "none" column

    def per_class_table(self):
        rows = []
        for i, c in enumerate(self.classes):
            tp, fp, fn = int(self.tp[i]), int(self.fp[i]), int(self.fn[i])
            precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            if np.isnan(precision) or np.isnan(recall):
                f1 = float('nan')
            elif precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            rows.append({'Category': c, 'TP': tp, 'FP': fp, 'FN': fn,
                         'Precision': precision, 'Recall': recall, 'F1': f1})
        return rows

    def macro(self):
        rows = self.per_class_table()
        precisions = [r['Precision'] for r in rows if not np.isnan(r['Precision'])]
        recalls = [r['Recall'] for r in rows if not np.isnan(r['Recall'])]
        f1s = [r['F1'] for r in rows if not np.isnan(r['F1'])]
        return {
            'Precision_macro': float(np.mean(precisions)) if precisions else float('nan'),
            'Recall_macro': float(np.mean(recalls)) if recalls else float('nan'),
            'F1_macro': float(np.mean(f1s)) if f1s else float('nan'),
        }

    def micro(self):
        tp, fp, fn = int(self.tp.sum()), int(self.fp.sum()), int(self.fn.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        return {'Precision_micro': precision, 'Recall_micro': recall}

    def confusion_df(self):
        cols = self.classes + ['none']
        return pd.DataFrame(self.matrix, index=self.classes, columns=cols)


# --- YOLOv5 ---
# Trained via train_yolov5.py using the `ultralytics` pip package (no repo
# clone needed) -> leave YOLOV5_REPO_PATH as None and just point at best.pt.
# (If you instead cloned github.com/ultralytics/yolov5 and trained with its
# train.py directly, set YOLOV5_REPO_PATH to that clone's path instead.)
YOLOV5_REPO_PATH = None
YOLOV5_WEIGHTS   = "yolo_runs/sign_detector/weights/best.pt"  # <-- EDIT if different
YOLOV5_CONF_THRES = 0.25
YOLOV5_IOU_THRES  = 0.45

# --- Faster R-CNN (torchvision) ---
FRCNN_WEIGHTS    = "faster_rcnn_sign_detector/best_frcnn.pth"  # <-- EDIT if different
FRCNN_NUM_CLASSES = 5       # background + 4 sign classes (crosswalk, speedlimit, stop, trafficlight)
FRCNN_CONF_THRES  = 0.5

print(f"Device : {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# OCCLUSION-LEVEL DETECTION -- same logic/config point as
# evaluate_baseline_models.py. Edit get_level() if it doesn't match
# your dataset's naming convention.
# ═══════════════════════════════════════════════════════════════

_LEVEL_PATTERNS = [
    re.compile(r'[/\\](\d{1,3})[/\\]'),
    re.compile(r'lvl[_\-]?(\d{1,3})', re.IGNORECASE),
    re.compile(r'level[_\-]?(\d{1,3})', re.IGNORECASE),
    re.compile(r'(\d{1,3})pct', re.IGNORECASE),
    re.compile(r'_(\d{1,3})_'),
]

def get_level(sample):
    if 'occ_level' in sample:
        return int(sample['occ_level'])
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
        print(f"  WARNING: {unmatched} samples had no detectable level -- check get_level()")
    return groups


def load_split(path):
    with open(path) as f:
        split = json.load(f)
    return split.get("test", split.get("val", []))


# ═══════════════════════════════════════════════════════════════
# YOLOv5 WRAPPER
# ═══════════════════════════════════════════════════════════════

class YoloDetector:
    """
    Supports two loading paths:
      - repo_path set to a local github.com/ultralytics/yolov5 clone -> old
        torch.hub 'custom' API (weights trained via that repo's train.py)
      - repo_path=None (or "ultralytics") -> the pip-installable `ultralytics`
        package's YOLO() API (weights trained via train_yolov5.py's default
        path, i.e. `pip install ultralytics` with no repo clone needed)
    """
    def __init__(self, repo_path, weights_path, conf=0.25, iou=0.45):
        self.backend = None

        if repo_path and os.path.isdir(repo_path):
            self.model = torch.hub.load(repo_path, 'custom', path=weights_path,
                                         source='local')
            self.model.conf = conf
            self.model.iou = iou
            self.model.to(DEVICE)
            self.model.eval()
            self.backend = 'hub'
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.conf = conf
            self.iou = iou
            self.backend = 'ultralytics'

    @torch.no_grad()
    def predict_classes(self, img_rgb_uint8):
        """Returns a list of 0-indexed predicted class ids for every box above
        the confidence threshold (empty list if nothing detected). This is the
        basis for proper precision/recall -- not just a detected/not-detected
        boolean."""
        if self.backend == 'hub':
            results = self.model(img_rgb_uint8, size=max(img_rgb_uint8.shape[:2]))
            preds = results.xyxy[0]
            if preds.shape[0] == 0:
                return []
            return preds[:, 5].cpu().numpy().astype(int).tolist()
        else:
            results = self.model.predict(img_rgb_uint8, conf=self.conf, iou=self.iou,
                                          verbose=False)
            boxes = results[0].boxes
            if len(boxes) == 0:
                return []
            return boxes.cls.cpu().numpy().astype(int).tolist()

    @torch.no_grad()
    def detect(self, img_rgb_uint8, expected_class_id=None):
        """
        img_rgb_uint8: HxWx3 uint8 RGB numpy array.
        expected_class_id: 0-indexed class id matching prepare_detector_dataset.py's
            CLASSES order (crosswalk=0, speedlimit=1, stop=2, trafficlight=3).
            If given, "detected" means a box above threshold with the CORRECT
            class -- not just any box. This matters a lot here: since training
            boxes are near full-frame, "any box exists" is almost always true
            regardless of occlusion, so it doesn't actually measure whether the
            sign is still recognizable. If None, falls back to any-box-present
            (legacy behaviour).

        NOTE: for proper precision/recall (not just a hit-rate), use
        predict_classes() + compute_confusion() instead of this method.
        """
        preds = self.predict_classes(img_rgb_uint8)
        if not preds:
            return False
        if expected_class_id is None:
            return True
        return expected_class_id in preds


# ═══════════════════════════════════════════════════════════════
# FASTER R-CNN WRAPPER (torchvision)
# ═══════════════════════════════════════════════════════════════

class FrcnnDetector:
    def __init__(self, weights_path, num_classes=2, conf=0.5):
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        state = torch.load(weights_path, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()
        self.conf = conf

    @torch.no_grad()
    def predict_classes(self, img_rgb_uint8):
        """Returns a list of 0-indexed predicted class ids for every box above
        the confidence threshold (FRCNN's 1-indexed labels converted back to
        0-indexed to match prepare_detector_dataset.py's CLASSES order)."""
        tensor = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)
        out = self.model(tensor)[0]
        scores = out['scores'].cpu().numpy()
        above = scores >= self.conf
        if not above.any():
            return []
        labels = out['labels'].cpu().numpy()[above]
        return (labels - 1).tolist()   # label 0 is background; shift back to 0-indexed

    @torch.no_grad()
    def detect(self, img_rgb_uint8, expected_class_id=None):
        """
        img_rgb_uint8: HxWx3 uint8 RGB numpy array.
        expected_class_id: 0-indexed class id (crosswalk=0, speedlimit=1,
            stop=2, trafficlight=3). See YoloDetector.detect() for why this
            class check matters.

        NOTE: for proper precision/recall (not just a hit-rate), use
        predict_classes() + compute_confusion() instead of this method.
        """
        preds = self.predict_classes(img_rgb_uint8)
        if not preds:
            return False
        if expected_class_id is None:
            return True
        return expected_class_id in preds


# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY -- A-I (loads the classes from your two training files)
# ═══════════════════════════════════════════════════════════════

def build_model_registry():
    """Import architectures lazily so this script can still be inspected /
    partially used even if one of the two training files isn't present yet."""
    from train_all_models import UNet, AdvancedUNet
    from train_baseline_models import (
        PlainAutoencoder, CEGenerator, PConvUNet, ViTInpainter, TARGET_SIZE as _TS
    )

    registry = [
        {"name": "Model A (Base U-Net)",      "arch": "unet",     "in_ch": 3,
         "ckpt": "model_A_base_unet/best_model_A.pth"},
        {"name": "Model B (Mask U-Net)",      "arch": "unet",     "in_ch": 5,
         "ckpt": "model_B_mask_unet/best_model_B.pth"},
        {"name": "Model C (ROI U-Net)",       "arch": "unet",     "in_ch": 5,
         "ckpt": "model_C_roi_unet/best_model_C.pth"},
        {"name": "Model D (Enhanced U-Net)",  "arch": "unet",     "in_ch": 5,
         "ckpt": "model_D_enhanced_unet/best_model_D.pth"},
        {"name": "Model E (Advanced U-Net)",  "arch": "advanced", "in_ch": 5,
         "ckpt": "model_E_advanced_unet/best_model_E.pth"},
        {"name": "Model F (Plain Autoencoder)", "arch": "plain_ae", "in_ch": 5,
         "ckpt": "model_F_plain_autoencoder/best_model_F.pth"},
        {"name": "Model G (Context-Encoder GAN)", "arch": "gan",   "in_ch": 5,
         "ckpt": "model_G_context_encoder_gan/best_model_G.pth"},
        {"name": "Model H (Partial-Conv U-Net)", "arch": "pconv", "in_ch": 5,
         "ckpt": "model_H_partial_conv_unet/best_model_H.pth"},
        {"name": "Model I (ViT Inpainter)",   "arch": "vit",      "in_ch": 5,
         "ckpt": "model_I_vit_inpainter/best_model_I.pth"},
    ]

    classes = {
        "unet": UNet, "advanced": AdvancedUNet, "plain_ae": PlainAutoencoder,
        "gan": CEGenerator, "pconv": PConvUNet, "vit": ViTInpainter,
    }
    return registry, classes


def load_inpaint_model(cfg, classes):
    arch = cfg["arch"]
    ckpt = torch.load(cfg["ckpt"], map_location=DEVICE)
    cls = classes[arch]

    if arch == "vit":
        model = cls(in_ch=cfg["in_ch"], out_ch=3, img_size=TARGET_SIZE[0]).to(DEVICE)
    else:
        model = cls(in_ch=cfg["in_ch"], out_ch=3).to(DEVICE)

    if arch == "gan":
        model.load_state_dict(ckpt["generator_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model


@torch.no_grad()
def run_inpaint(model, cfg, x, occ_mask):
    arch = cfg["arch"]
    xin = x[:, :cfg["in_ch"]]
    if arch == "pconv":
        valid_mask = 1.0 - occ_mask
        return model(xin, valid_mask)
    if arch == "advanced":
        out = model(xin, return_aux=False)
        return out
    return model(xin)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (single sample, no augmentation, matches training scripts)
# ═══════════════════════════════════════════════════════════════

def load_rgb(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def load_mask(path, size):
    if not path or not os.path.exists(path):
        return np.zeros((size[1], size[0]), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)

def build_input_tensor(sample, size):
    occ = np.clip(load_rgb(sample['occ_path'], size), 0, 1)
    mask = np.clip(load_mask(sample.get('mask_path', ''), size), 0, 1)
    bbox_mask = np.clip(cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1), 0, 1)
    x = np.concatenate([occ, mask[..., None], bbox_mask[..., None]], axis=-1)
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)
    m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    return x, m, occ


def tensor_to_uint8(t):
    """t: (1,3,H,W) in [0,1] -> HxWx3 uint8 RGB."""
    arr = t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading detectors...")
    yolo = YoloDetector(YOLOV5_REPO_PATH, YOLOV5_WEIGHTS, YOLOV5_CONF_THRES, YOLOV5_IOU_THRES)
    frcnn = FrcnnDetector(FRCNN_WEIGHTS, FRCNN_NUM_CLASSES, FRCNN_CONF_THRES)

    print("Loading test split...")
    test_samples = load_split(SPLIT_PATH)

    # Level 0 (clean, no occlusion) lives in dataset/actual/<category>/ --
    # there is no dataset/occluded/0/ folder on disk, unlike levels 10-80
    # which do have their own occluded/<level>/ folders. So we synthesize
    # level-0 rows by pointing occ_path at the clean image itself (mask
    # empty -> nothing occluded). Same logic as PIPLINEdetect.py.
    seen = {}
    for s in test_samples:
        key = s.get('image_id', s['clean_path'])
        seen[key] = s
    level0_samples = []
    for s in seen.values():
        level0_samples.append({
            'image_id': s.get('image_id', os.path.splitext(os.path.basename(s['clean_path']))[0]),
            'category': s.get('category', 'unk'),
            'occ_level': 0, 'clean_path': s['clean_path'],
            'occ_path': s['clean_path'], 'mask_path': '',
        })
    test_samples = test_samples + level0_samples
    print(f"  Synthesized {len(level0_samples)} clean (0%) samples from dataset/actual/")

    level_groups = group_by_level(test_samples)
    for lvl, s in level_groups.items():
        print(f"  Level {lvl:>3}% : {len(s)} samples")

    registry, classes_reg = build_model_registry()

    # ── pass 1: pre-inpaint (occluded image) detection, shared across all models ──
    print("\nRunning pre-inpaint detection on raw occluded images...")
    preinpaint = {}   # level -> {'yolo': ConfusionAccumulator, 'frcnn': ConfusionAccumulator}
    for lvl in OCCLUSION_LEVELS:
        samples = level_groups.get(lvl, [])
        if not samples:
            continue
        acc_yolo = ConfusionAccumulator(CLASSES)
        acc_frcnn = ConfusionAccumulator(CLASSES)
        for s in samples:
            occ_uint8 = (load_rgb(s['occ_path'], TARGET_SIZE) * 255).astype(np.uint8)
            cid = class_id_for(s)
            acc_yolo.update(cid, yolo.predict_classes(occ_uint8))
            acc_frcnn.update(cid, frcnn.predict_classes(occ_uint8))
        preinpaint[lvl] = {'yolo': acc_yolo, 'frcnn': acc_frcnn}
        my, mf = acc_yolo.macro(), acc_frcnn.macro()
        print(f"  Level {lvl:>3}%  YOLO P={my['Precision_macro']:.3f} R={my['Recall_macro']:.3f}  "
              f"FRCNN P={mf['Precision_macro']:.3f} R={mf['Recall_macro']:.3f}")

    # ── pass 2: per-model inpainting + post-inpaint detection ──
    all_rows = []
    confusion_rows = []   # per-category breakdown, every (model, level, detector, pre/post)
    for cfg in registry:
        if not os.path.exists(cfg["ckpt"]):
            print(f"\nSkipping {cfg['name']} -- checkpoint not found at {cfg['ckpt']}")
            continue

        print(f"\n{'='*64}\n  {cfg['name']}\n{'='*64}")
        model = load_inpaint_model(cfg, classes_reg)

        for lvl in OCCLUSION_LEVELS:
            samples = level_groups.get(lvl, [])
            if not samples:
                continue

            acc_yolo = ConfusionAccumulator(CLASSES)
            acc_frcnn = ConfusionAccumulator(CLASSES)
            t0 = time.time()
            for s in samples:
                x, occ_mask, _ = build_input_tensor(s, TARGET_SIZE)
                x, occ_mask = x.to(DEVICE), occ_mask.to(DEVICE)
                pred = run_inpaint(model, cfg, x, occ_mask)
                pred_uint8 = tensor_to_uint8(pred)

                cid = class_id_for(s)
                acc_yolo.update(cid, yolo.predict_classes(pred_uint8))
                acc_frcnn.update(cid, frcnn.predict_classes(pred_uint8))
            elapsed = time.time() - t0
            n = len(samples)

            pre_yolo_macro = preinpaint[lvl]['yolo'].macro()
            pre_frcnn_macro = preinpaint[lvl]['frcnn'].macro()
            post_yolo_macro = acc_yolo.macro()
            post_frcnn_macro = acc_frcnn.macro()

            row = {
                'Model': cfg['name'], 'Level': f"{lvl}%", 'N': n,
                'YOLO_Precision_pre': pre_yolo_macro['Precision_macro'],
                'YOLO_Recall_pre': pre_yolo_macro['Recall_macro'],
                'YOLO_Precision_post': post_yolo_macro['Precision_macro'],
                'YOLO_Recall_post': post_yolo_macro['Recall_macro'],
                'YOLO_Recall_gain': post_yolo_macro['Recall_macro'] - pre_yolo_macro['Recall_macro'],
                'FRCNN_Precision_pre': pre_frcnn_macro['Precision_macro'],
                'FRCNN_Recall_pre': pre_frcnn_macro['Recall_macro'],
                'FRCNN_Precision_post': post_frcnn_macro['Precision_macro'],
                'FRCNN_Recall_post': post_frcnn_macro['Recall_macro'],
                'FRCNN_Recall_gain': post_frcnn_macro['Recall_macro'] - pre_frcnn_macro['Recall_macro'],
                'Total_sec': elapsed, 'Avg_ms': (elapsed / n) * 1000 if n else float('nan'),
            }
            all_rows.append(row)

            for det_name, acc_pre, acc_post in [('YOLO', preinpaint[lvl]['yolo'], acc_yolo),
                                                 ('FRCNN', preinpaint[lvl]['frcnn'], acc_frcnn)]:
                for pc in acc_post.per_class_table():
                    confusion_rows.append({
                        'Model': cfg['name'], 'Level': f"{lvl}%", 'Detector': det_name,
                        'Category': pc['Category'], 'TP': pc['TP'], 'FP': pc['FP'], 'FN': pc['FN'],
                        'Precision': pc['Precision'], 'Recall': pc['Recall'], 'F1': pc['F1'],
                    })

            print(f"  Level {lvl:>3}% (n={n:>3})  "
                  f"YOLO P/R: {pre_yolo_macro['Precision_macro']:.3f}/{pre_yolo_macro['Recall_macro']:.3f} "
                  f"-> {post_yolo_macro['Precision_macro']:.3f}/{post_yolo_macro['Recall_macro']:.3f}   "
                  f"FRCNN P/R: {pre_frcnn_macro['Precision_macro']:.3f}/{pre_frcnn_macro['Recall_macro']:.3f} "
                  f"-> {post_frcnn_macro['Precision_macro']:.3f}/{post_frcnn_macro['Recall_macro']:.3f}")

    if not all_rows:
        print("\nNo models were evaluated -- check checkpoint paths in build_model_registry().")
        return

    full_df = pd.DataFrame(all_rows)
    conf_df = pd.DataFrame(confusion_rows)
    with pd.ExcelWriter("detection_results.xlsx", engine="openpyxl") as writer:
        full_df.to_excel(writer, sheet_name="Detection Results", index=False)
        conf_df.to_excel(writer, sheet_name="Per-category P R F1", index=False)
    print("\nSaved -> detection_results.xlsx (2 sheets)")

    overview = (full_df.groupby('Model')[
        ['YOLO_Precision_pre', 'YOLO_Recall_pre', 'YOLO_Precision_post', 'YOLO_Recall_post',
         'YOLO_Recall_gain', 'FRCNN_Precision_pre', 'FRCNN_Recall_pre',
         'FRCNN_Precision_post', 'FRCNN_Recall_post', 'FRCNN_Recall_gain']
    ].mean().reset_index())
    overview.to_csv("detection_results_summary.csv", index=False)

    print(f"\n{'='*64}")
    print("  DETECTION RESULTS -- OVERVIEW (macro P/R, mean across all occlusion levels)")
    print(f"{'='*64}")
    print(overview.to_string(index=False))
    print("\nSaved -> detection_results_summary.csv")



if __name__ == "__main__":
    main()
