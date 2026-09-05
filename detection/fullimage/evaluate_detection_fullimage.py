"""
evaluate_detection_fullimage.py
==================================
Evaluates YOLOv5 + Faster R-CNN on full-scene images, using REAL
multi-object ground truth (from the XML annotations) with proper
IoU-based matching -- not the "does the right class appear anywhere"
simplification used for the cropped-patch pipeline. This is a genuine
(if simplified, single-IoU-threshold) object-detection evaluation.

For each of the 9 inpainting models and each occlusion level, this
compares detection performance on:
  - the clean image (once, shared baseline / ceiling)
  - the raw occluded image (once per level, shared across models)
  - that model's inpainted output

BEFORE RUNNING -- configure these:
    YOLOV5_WEIGHTS   -- from train_yolov5_fullimage.py
    FRCNN_WEIGHTS    -- from train_faster_rcnn_fullimage.py
    get_level()      -- same customization point as the cropped-patch
                        pipeline's evaluate_detection.py, if your
                        occ_level field/paths differ

Usage:
    python evaluate_detection_fullimage.py
"""

import os, re, json, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

SPLIT_PATH   = os.path.join("traintestsplit", "data_split.json")
ANNOT_DIR    = "e"
TARGET_SIZE  = (256, 256)     # inpainting models' working resolution
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OCCLUSION_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80]

CLASSES = ["crosswalk", "speedlimit", "stop", "trafficlight"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
IOU_THRESHOLD = 0.5

YOLOV5_WEIGHTS = "yolo_runs_fullimage/sign_detector_fullimage/weights/best.pt"
YOLOV5_CONF_THRES = 0.25
YOLOV5_IOU_THRES  = 0.45

FRCNN_WEIGHTS = "faster_rcnn_fullimage_detector/best_frcnn_fullimage.pth"
FRCNN_NUM_CLASSES = len(CLASSES) + 1
FRCNN_CONF_THRES = 0.5

OUTPUT_ROOT = "detection_results_fullimage"

print(f"Device : {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# OCCLUSION-LEVEL DETECTION (same customization point as before)
# ═══════════════════════════════════════════════════════════════

_LEVEL_PATTERNS = [
    re.compile(r'[/\\](\d{1,3})[/\\]'), re.compile(r'lvl[_\-]?(\d{1,3})', re.IGNORECASE),
    re.compile(r'level[_\-]?(\d{1,3})', re.IGNORECASE), re.compile(r'(\d{1,3})pct', re.IGNORECASE),
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
    for s in samples:
        lvl = get_level(s)
        if lvl is not None:
            groups.setdefault(lvl, []).append(s)
    return groups

def load_split(path):
    with open(path) as f:
        split = json.load(f)
    return split.get("test", split.get("val", []))


# ═══════════════════════════════════════════════════════════════
# GROUND TRUTH (real XML boxes, in ORIGINAL image pixel space)
# ═══════════════════════════════════════════════════════════════

def parse_ground_truth(clean_path):
    """Returns list of (class_id, x1, y1, x2, y2) in original pixel space."""
    base = os.path.splitext(os.path.basename(clean_path))[0]
    xml_path = os.path.join(ANNOT_DIR, base + ".xml")
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        out = []
        for obj in root.findall("object"):
            name = obj.find("name"); bb = obj.find("bndbox")
            if name is None or bb is None:
                continue
            cls = name.text.strip().lower()
            if cls not in CLASS_TO_ID:
                continue
            x1 = int(float(bb.find("xmin").text)); y1 = int(float(bb.find("ymin").text))
            x2 = int(float(bb.find("xmax").text)); y2 = int(float(bb.find("ymax").text))
            if x2 > x1 and y2 > y1:
                out.append((CLASS_TO_ID[cls], x1, y1, x2, y2))
        return out
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
# IoU-BASED MULTI-OBJECT MATCHING
# ═══════════════════════════════════════════════════════════════

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionAccumulator:
    """
    Real multi-object precision/recall via greedy IoU matching (single
    IoU threshold, per-class), the standard "AP@0.5-style" matching
    procedure -- not full COCO mAP (no PR-curve integration), but a
    genuine box-level TP/FP/FN count, unlike the cropped-patch
    pipeline's "does the class appear anywhere" simplification.
    """
    def __init__(self, classes, iou_thresh=IOU_THRESHOLD):
        self.classes = classes
        self.iou_thresh = iou_thresh
        n = len(classes)
        self.tp = np.zeros(n, dtype=int)
        self.fp = np.zeros(n, dtype=int)
        self.fn = np.zeros(n, dtype=int)

    def update(self, gt_objs, pred_objs):
        """
        gt_objs:   list of (class_id, x1, y1, x2, y2)
        pred_objs: list of (class_id, score, x1, y1, x2, y2)
        """
        for cls_id in range(len(self.classes)):
            gt_this = [g[1:] for g in gt_objs if g[0] == cls_id]
            preds_this = sorted(
                [p for p in pred_objs if p[0] == cls_id], key=lambda p: -p[1])
            matched = [False] * len(gt_this)

            for _, score, x1, y1, x2, y2 in preds_this:
                best_iou, best_j = 0.0, -1
                for j, gbox in enumerate(gt_this):
                    if matched[j]:
                        continue
                    iou = iou_xyxy((x1, y1, x2, y2), gbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= self.iou_thresh and best_j >= 0:
                    matched[best_j] = True
                    self.tp[cls_id] += 1
                else:
                    self.fp[cls_id] += 1

            self.fn[cls_id] += matched.count(False)

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
        p = [r['Precision'] for r in rows if not np.isnan(r['Precision'])]
        r_ = [r['Recall'] for r in rows if not np.isnan(r['Recall'])]
        f1 = [r['F1'] for r in rows if not np.isnan(r['F1'])]
        return {'Precision_macro': float(np.mean(p)) if p else float('nan'),
                'Recall_macro': float(np.mean(r_)) if r_ else float('nan'),
                'F1_macro': float(np.mean(f1)) if f1 else float('nan')}


# ═══════════════════════════════════════════════════════════════
# DETECTOR WRAPPERS (return full boxes, not just class presence)
# ═══════════════════════════════════════════════════════════════

class YoloDetectorFull:
    def __init__(self, weights_path, conf=0.25, iou=0.45):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf, self.iou = conf, iou

    def detect(self, img_rgb_uint8):
        """Returns list of (class_id, score, x1, y1, x2, y2) in the
        SAME pixel space as img_rgb_uint8."""
        results = self.model.predict(img_rgb_uint8, conf=self.conf, iou=self.iou, verbose=False)
        boxes = results[0].boxes
        out = []
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            out.append((cls, score, x1, y1, x2, y2))
        return out


class FrcnnDetectorFull:
    def __init__(self, weights_path, num_classes, conf=0.5):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_f = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
        self.conf = conf

    @torch.no_grad()
    def detect(self, img_rgb_uint8):
        tensor = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)
        out = self.model(tensor)[0]
        boxes = out['boxes'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score < self.conf:
                continue
            cls_id = int(label) - 1   # shift back to 0-indexed
            if cls_id < 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append((cls_id, float(score), x1, y1, x2, y2))
        return results


# ═══════════════════════════════════════════════════════════════
# INPAINTING MODEL LOADING (reuses evaluate_all_models.py's classes)
# ═══════════════════════════════════════════════════════════════

def build_inpaint_registry():
    import evaluate_all_models as ev
    registry = [
        {"name": "Model A (Base U-Net)",       "arch": "unet",     "in_ch": 3,
         "ckpt": os.path.join("inpainting_model", "best_inpainting.pth")},
        {"name": "Model B (Mask U-Net)",       "arch": "unet",     "in_ch": 5,
         "ckpt": os.path.join("model_B_mask_unet", "best_mask_unet.pth")},
        {"name": "Model C (ROI U-Net)",        "arch": "unet",     "in_ch": 5,
         "ckpt": os.path.join("model_C_roi_unet", "best_roi_unet.pth")},
        {"name": "Model D (Enhanced U-Net)",   "arch": "unet",     "in_ch": 5,
         "ckpt": os.path.join("enhanced_unetapril2026", "best_roi_unet_ssim.pth")},
        {"name": "Model E (Advanced U-Net)",   "arch": "advanced", "in_ch": 5,
         "ckpt": os.path.join("model_E_advanced_unet", "best_advanced_unet.pth")},
        {"name": "Model F (Plain Autoencoder)", "arch": "plain_ae", "in_ch": 5,
         "ckpt": os.path.join("model_F_plain_autoencoder", "best_model_F.pth")},
        {"name": "Model G (Context-Encoder GAN)", "arch": "gan",   "in_ch": 5,
         "ckpt": os.path.join("model_G_context_encoder_gan", "best_model_G.pth")},
        {"name": "Model H (Partial-Conv U-Net)", "arch": "pconv", "in_ch": 5,
         "ckpt": os.path.join("model_H_partial_conv_unet", "best_model_H.pth")},
        {"name": "Model I (ViT Inpainter)",    "arch": "vit",      "in_ch": 5,
         "ckpt": os.path.join("model_I_vit_inpainter", "best_model_I.pth")},
    ]
    return registry, ev


def load_inpaint_model(cfg, ev):
    ckpt = torch.load(cfg["ckpt"], map_location=DEVICE, weights_only=False)
    in_ch = ckpt.get("in_ch", cfg["in_ch"])
    arch = ckpt.get("arch", cfg["arch"])

    if arch == "advanced":
        model = ev.AdvancedUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    elif arch == "plain_ae":
        model = ev.PlainAutoencoder(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    elif arch == "gan":
        model = ev.CEGenerator(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt["generator_state_dict"])
    elif arch == "pconv":
        model = ev.PConvUNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    elif arch == "vit":
        model = ev.ViTInpainter(in_ch=in_ch, out_ch=3, img_size=TARGET_SIZE[0]).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = ev.UNet(in_ch=in_ch, out_ch=3).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model, arch, in_ch


@torch.no_grad()
def run_inpaint(model, arch, in_ch, x256, occ_mask256):
    xin = x256[:, :in_ch]
    if arch == "pconv":
        valid_mask = 1.0 - occ_mask256
        return model(xin, valid_mask)
    if arch == "advanced":
        return model(xin, return_aux=False)
    return model(xin)


# ═══════════════════════════════════════════════════════════════
# IMAGE / MASK LOADING HELPERS
# ═══════════════════════════════════════════════════════════════

def load_rgb_native(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_rgb_256(path):
    img = load_rgb_native(path)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def load_mask_256(path):
    if not path or not os.path.exists(path):
        return np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)

def build_5ch_input(sample):
    occ = np.clip(load_rgb_256(sample['occ_path']), 0, 1)
    mask = np.clip(load_mask_256(sample.get('mask_path', '')), 0, 1)
    bbox_mask = np.clip(cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1), 0, 1)
    x = np.concatenate([occ, mask[..., None], bbox_mask[..., None]], axis=-1)
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)
    m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    return x, m

def tensor_to_native_uint8(pred256, native_hw):
    """Upsamples the model's 256x256 output back to the image's native
    resolution, since ground-truth boxes are in native pixel space."""
    arr = pred256[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr_uint8 = (arr * 255).astype(np.uint8)
    return cv2.resize(arr_uint8, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Loading detectors...")
    yolo = YoloDetectorFull(YOLOV5_WEIGHTS, YOLOV5_CONF_THRES, YOLOV5_IOU_THRES)
    frcnn = FrcnnDetectorFull(FRCNN_WEIGHTS, FRCNN_NUM_CLASSES, FRCNN_CONF_THRES)

    print("Loading test split...")
    test_samples = load_split(SPLIT_PATH)

    # synthesize clean/0% baseline (occ_path = clean_path, no mask)
    seen = {}
    for s in test_samples:
        seen[s.get('image_id', s['clean_path'])] = s
    level0 = []
    for s in seen.values():
        level0.append({'image_id': s.get('image_id', s['clean_path']), 'clean_path': s['clean_path'],
                       'occ_path': s['clean_path'], 'mask_path': '', 'occ_level': 0})
    test_samples = test_samples + level0

    level_groups = group_by_level(test_samples)
    for lvl, s in level_groups.items():
        print(f"  Level {lvl:>3}% : {len(s)} samples")

    registry, ev = build_inpaint_registry()

    # ── standalone clean-image detection baseline ──
    print("\nRunning detection on the actual clean images (no occlusion)...")
    acc_yolo_clean = DetectionAccumulator(CLASSES)
    acc_frcnn_clean = DetectionAccumulator(CLASSES)
    for s in seen.values():
        native = load_rgb_native(s['clean_path'])
        gt = parse_ground_truth(s['clean_path'])
        acc_yolo_clean.update(gt, yolo.detect(native))
        acc_frcnn_clean.update(gt, frcnn.detect(native))
    cym, cfm = acc_yolo_clean.macro(), acc_frcnn_clean.macro()
    print(f"  YOLOv5:       P={cym['Precision_macro']:.3f}  R={cym['Recall_macro']:.3f}")
    print(f"  Faster R-CNN: P={cfm['Precision_macro']:.3f}  R={cfm['Recall_macro']:.3f}")
    clean_rows = []
    for det_name, acc in [('YOLO', acc_yolo_clean), ('FRCNN', acc_frcnn_clean)]:
        for pc in acc.per_class_table():
            clean_rows.append({'Detector': det_name, **pc})

    # ── pre-inpaint detection on raw occluded images, per level (shared across models) ──
    print("\nRunning pre-inpaint detection on occluded images...")
    preinpaint = {}
    for lvl in OCCLUSION_LEVELS:
        samples = level_groups.get(lvl, [])
        if not samples:
            continue
        acc_y, acc_f = DetectionAccumulator(CLASSES), DetectionAccumulator(CLASSES)
        for s in samples:
            native = load_rgb_native(s['occ_path'])
            gt = parse_ground_truth(s['clean_path'])
            acc_y.update(gt, yolo.detect(native))
            acc_f.update(gt, frcnn.detect(native))
        preinpaint[lvl] = {'yolo': acc_y, 'frcnn': acc_f}
        my, mf = acc_y.macro(), acc_f.macro()
        print(f"  Level {lvl:>3}%  YOLO P={my['Precision_macro']:.3f} R={my['Recall_macro']:.3f}  "
              f"FRCNN P={mf['Precision_macro']:.3f} R={mf['Recall_macro']:.3f}")

    # ── per-model inpainting + post-inpaint detection ──
    all_rows = []
    for cfg in registry:
        if not os.path.exists(cfg["ckpt"]):
            print(f"\nSkipping {cfg['name']} -- checkpoint not found at {cfg['ckpt']}")
            continue

        print(f"\n{'='*64}\n  {cfg['name']}\n{'='*64}")
        model, arch, in_ch = load_inpaint_model(cfg, ev)

        for lvl in OCCLUSION_LEVELS:
            samples = level_groups.get(lvl, [])
            if not samples:
                continue

            acc_y, acc_f = DetectionAccumulator(CLASSES), DetectionAccumulator(CLASSES)
            t0 = time.time()
            for s in samples:
                native_img = load_rgb_native(s['clean_path'])
                native_hw = native_img.shape[:2]
                gt = parse_ground_truth(s['clean_path'])

                x, occ_mask = build_5ch_input(s)
                x, occ_mask = x.to(DEVICE), occ_mask.to(DEVICE)
                pred256 = run_inpaint(model, arch, in_ch, x, occ_mask)
                pred_native = tensor_to_native_uint8(pred256, native_hw)

                acc_y.update(gt, yolo.detect(pred_native))
                acc_f.update(gt, frcnn.detect(pred_native))
            elapsed = time.time() - t0
            n = len(samples)

            pre_y, pre_f = preinpaint[lvl]['yolo'].macro(), preinpaint[lvl]['frcnn'].macro()
            post_y, post_f = acc_y.macro(), acc_f.macro()

            row = {
                'Model': cfg['name'], 'Level': f"{lvl}%", 'N': n,
                'YOLO_Precision_pre': pre_y['Precision_macro'], 'YOLO_Recall_pre': pre_y['Recall_macro'],
                'YOLO_Precision_post': post_y['Precision_macro'], 'YOLO_Recall_post': post_y['Recall_macro'],
                'YOLO_Recall_gain': post_y['Recall_macro'] - pre_y['Recall_macro'],
                'FRCNN_Precision_pre': pre_f['Precision_macro'], 'FRCNN_Recall_pre': pre_f['Recall_macro'],
                'FRCNN_Precision_post': post_f['Precision_macro'], 'FRCNN_Recall_post': post_f['Recall_macro'],
                'FRCNN_Recall_gain': post_f['Recall_macro'] - pre_f['Recall_macro'],
                'Total_sec': elapsed, 'Avg_ms': (elapsed / n) * 1000 if n else float('nan'),
            }
            all_rows.append(row)
            print(f"  Level {lvl:>3}% (n={n:>3})  "
                  f"YOLO P/R: {pre_y['Precision_macro']:.3f}/{pre_y['Recall_macro']:.3f} -> "
                  f"{post_y['Precision_macro']:.3f}/{post_y['Recall_macro']:.3f}   "
                  f"FRCNN P/R: {pre_f['Precision_macro']:.3f}/{pre_f['Recall_macro']:.3f} -> "
                  f"{post_f['Precision_macro']:.3f}/{post_f['Recall_macro']:.3f}")

    if not all_rows:
        print("\nNo models were evaluated.")
        return

    full_df = pd.DataFrame(all_rows)
    clean_df = pd.DataFrame(clean_rows)
    xlsx_path = os.path.join(OUTPUT_ROOT, "detection_results_fullimage.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        clean_df.to_excel(writer, sheet_name="Clean Image Baseline", index=False)
        full_df.to_excel(writer, sheet_name="Detection Results", index=False)
    print(f"\nSaved -> {xlsx_path}")

    overview = full_df.groupby('Model')[
        ['YOLO_Precision_pre', 'YOLO_Recall_pre', 'YOLO_Precision_post', 'YOLO_Recall_post', 'YOLO_Recall_gain',
         'FRCNN_Precision_pre', 'FRCNN_Recall_pre', 'FRCNN_Precision_post', 'FRCNN_Recall_post', 'FRCNN_Recall_gain']
    ].mean().reset_index()
    overview.to_csv(os.path.join(OUTPUT_ROOT, "detection_summary_fullimage.csv"), index=False)

    print(f"\n{'='*64}\n  OVERVIEW (macro P/R, mean across levels)\n{'='*64}")
    print(overview.to_string(index=False))
    print(f"\nAll outputs saved under: {os.path.abspath(OUTPUT_ROOT)}/")


if __name__ == "__main__":
    main()
