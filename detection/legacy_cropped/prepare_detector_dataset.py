"""
prepare_detector_dataset.py
=============================
Builds detector-ready training data from the SAME clean images used to
train the U-Net inpainting models (dataset/actual/<category>/*.png).

Your data_split.json entries repeat each image once per occlusion level
(8x), so this script first dedupes back down to the underlying unique
images, then:

  1. Synthesizes a bounding box for each crop. These images are already
     tight single-object crops (that's what dataset/actual/ IS), so the
     box is the full image with a small inset margin -- there's no
     finer-grained box to recover. This is a deliberate, documented
     simplification: the detectors will learn "classify + localize
     the dominant object", which is exactly what you need for the
     downstream question ("is the sign still recognizable after
     inpainting?"), even though it won't generalize to multi-sign,
     full driving-scene images.

  2. Writes a YOLOv5-format dataset:
         yolo_dataset/images/train/*.png   yolo_dataset/labels/train/*.txt
         yolo_dataset/images/val/*.png     yolo_dataset/labels/val/*.txt
         yolo_dataset/data.yaml
  3. Writes a Faster R-CNN manifest (JSON) with the same split + boxes,
     consumed directly by train_faster_rcnn.py.

Usage:
    python prepare_detector_dataset.py
"""

import os, json, shutil
import cv2

SPLIT_PATH   = os.path.join("traintestsplit", "data_split.json")
YOLO_OUT_DIR = "yolo_dataset"
FRCNN_MANIFEST_TRAIN = "frcnn_manifest_train.json"
FRCNN_MANIFEST_VAL   = "frcnn_manifest_val.json"

# Must match the order used everywhere else (alphabetical, matching the
# Kaggle Road Sign Detection taxonomy your dataset/actual/ folders use).
CLASSES = ["crosswalk", "speedlimit", "stop", "trafficlight"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Inset margin for the synthesized "whole-crop" box (fraction of W/H).
BOX_MARGIN = 0.02


def dedupe_images(split_samples):
    """Collapse the (image x occlusion_level) rows back to unique images."""
    seen = {}
    for s in split_samples:
        seen[s['image_id']] = {'image_id': s['image_id'], 'category': s['category'],
                                'clean_path': s['clean_path']}
    return list(seen.values())


def synth_bbox(w, h, margin=BOX_MARGIN):
    """Full-frame box with a small inset margin, in pixel xyxy."""
    mx, my = int(w * margin), int(h * margin)
    return mx, my, w - mx, h - my


def build_yolo_dataset(train_items, val_items, out_dir=YOLO_OUT_DIR):
    for split_name, items in [("train", train_items), ("val", val_items)]:
        img_dir = os.path.join(out_dir, "images", split_name)
        lbl_dir = os.path.join(out_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        n_written = 0
        for item in items:
            src = item['clean_path']
            if not os.path.exists(src):
                continue
            img = cv2.imread(src)
            if img is None:
                continue
            h, w = img.shape[:2]
            x1, y1, x2, y2 = synth_bbox(w, h)

            # YOLO format: class cx cy w h, all normalized 0-1
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cls_id = CLASS_TO_ID[item['category']]

            fname = f"{item['image_id']}.png"
            dst_img = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst_img)

            lbl_path = os.path.join(lbl_dir, f"{item['image_id']}.txt")
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            n_written += 1

        print(f"  YOLO {split_name}: wrote {n_written}/{len(items)} images+labels")

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")
    print(f"  Wrote {yaml_path}")


def build_frcnn_manifest(items, path):
    rows = []
    for item in items:
        src = item['clean_path']
        if not os.path.exists(src):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = synth_bbox(w, h)
        rows.append({
            'image_id': item['image_id'],
            'clean_path': src,
            'category': item['category'],
            'label_id': CLASS_TO_ID[item['category']] + 1,  # +1: 0 is background
            'bbox': [x1, y1, x2, y2],
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Wrote {path} ({len(rows)} images)")


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    train_items = dedupe_images(split['train'])
    val_items   = dedupe_images(split['test'])
    print(f"Unique images -- train: {len(train_items)}  val: {len(val_items)}")

    print("\nBuilding YOLOv5-format dataset...")
    build_yolo_dataset(train_items, val_items)

    print("\nBuilding Faster R-CNN manifests...")
    build_frcnn_manifest(train_items, FRCNN_MANIFEST_TRAIN)
    build_frcnn_manifest(val_items, FRCNN_MANIFEST_VAL)

    print("\nDone. Classes (in order, matching label ids):")
    for i, c in enumerate(CLASSES):
        print(f"  YOLO class {i} / FRCNN label {i+1}: {c}")


if __name__ == "__main__":
    main()
