"""
prepare_detector_dataset_fullimage.py
========================================
Builds detector training data from the FULL (877) images using their
REAL PASCAL VOC XML annotations -- every object's actual class and
bounding box, not a synthesized full-frame box. Since these are full
scenes, a single image can (and often does) contain multiple objects
of different classes.

Run this BEFORE train_yolov5_fullimage.py / train_faster_rcnn_fullimage.py.

Output:
    yolo_dataset_fullimage/images/{train,val}/*.png
    yolo_dataset_fullimage/labels/{train,val}/*.txt   (one line per object)
    yolo_dataset_fullimage/data.yaml
    frcnn_manifest_fullimage_train.json   (list of {image, boxes:[[x1,y1,x2,y2],...], labels:[...]})
    frcnn_manifest_fullimage_val.json

Usage:
    python prepare_detector_dataset_fullimage.py
"""

import os, json, shutil
import cv2
import xml.etree.ElementTree as ET

SPLIT_PATH   = os.path.join("traintestsplit", "data_split.json")
ANNOT_DIR    = "e"
YOLO_OUT_DIR = "yolo_dataset_fullimage"
FRCNN_MANIFEST_TRAIN = "frcnn_manifest_fullimage_train.json"
FRCNN_MANIFEST_VAL   = "frcnn_manifest_fullimage_val.json"

# Kaggle Road Sign Detection's native 4 classes.
CLASSES = ["crosswalk", "speedlimit", "stop", "trafficlight"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def dedupe_images(split_samples):
    seen = {}
    for s in split_samples:
        key = s.get('image_id', s['clean_path'])
        seen[key] = s['clean_path']
    return list(seen.values())


def parse_annotation_with_class(xml_path):
    """Returns list of (class_name, xmin, ymin, xmax, ymax) in ORIGINAL
    image pixel coordinates (no resizing -- YOLO/FRCNN handle that
    internally). Unlike the inpainting pipeline's parse_annotation(),
    this keeps the class name since full images can have multiple
    different-class objects."""
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = []
        for obj in root.findall("object"):
            name = obj.find("name")
            bb = obj.find("bndbox")
            if name is None or bb is None:
                continue
            cls = name.text.strip().lower()
            xmin = int(float(bb.find("xmin").text))
            ymin = int(float(bb.find("ymin").text))
            xmax = int(float(bb.find("xmax").text))
            ymax = int(float(bb.find("ymax").text))
            if xmax > xmin and ymax > ymin:
                objs.append((cls, xmin, ymin, xmax, ymax))
        return objs
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")
        return []


def build_yolo_dataset(train_paths, val_paths, out_dir=YOLO_OUT_DIR):
    unknown_classes = set()
    for split_name, paths in [("train", train_paths), ("val", val_paths)]:
        img_dir = os.path.join(out_dir, "images", split_name)
        lbl_dir = os.path.join(out_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        n_written, n_objects = 0, 0
        for clean_path in paths:
            if not os.path.exists(clean_path):
                continue
            img = cv2.imread(clean_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            base = os.path.splitext(os.path.basename(clean_path))[0]
            xml_path = os.path.join(ANNOT_DIR, base + ".xml")
            objs = parse_annotation_with_class(xml_path)
            if not objs:
                continue   # skip images with no valid annotation

            fname = f"{base}.png"
            shutil.copyfile(clean_path, os.path.join(img_dir, fname))

            lines = []
            for cls, x1, y1, x2, y2 in objs:
                if cls not in CLASS_TO_ID:
                    unknown_classes.add(cls)
                    continue
                cls_id = CLASS_TO_ID[cls]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_objects += 1

            with open(os.path.join(lbl_dir, base + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_written += 1

        print(f"  YOLO {split_name}: {n_written}/{len(paths)} images, {n_objects} objects")

    if unknown_classes:
        print(f"  WARNING: found class names not in CLASSES list: {unknown_classes}")
        print(f"  -> add them to CLASSES at the top of this script and re-run.")

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")
    print(f"  Wrote {out_dir}/data.yaml")


def build_frcnn_manifest(paths, out_path):
    rows = []
    skipped_no_annot = 0
    for clean_path in paths:
        if not os.path.exists(clean_path):
            continue
        base = os.path.splitext(os.path.basename(clean_path))[0]
        xml_path = os.path.join(ANNOT_DIR, base + ".xml")
        objs = parse_annotation_with_class(xml_path)
        if not objs:
            skipped_no_annot += 1
            continue

        boxes, labels = [], []
        for cls, x1, y1, x2, y2 in objs:
            if cls not in CLASS_TO_ID:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[cls] + 1)   # +1: 0 reserved for background

        if not boxes:
            continue
        rows.append({'image_path': clean_path, 'boxes': boxes, 'labels': labels})

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Wrote {out_path} ({len(rows)} images, "
          f"{sum(len(r['boxes']) for r in rows)} objects, "
          f"{skipped_no_annot} skipped with no annotation)")


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    train_paths = dedupe_images(split['train'])
    val_paths   = dedupe_images(split['test'])
    print(f"Unique full images -- train: {len(train_paths)}  val: {len(val_paths)}")

    print("\nBuilding YOLOv5-format dataset (multi-object, multi-class)...")
    build_yolo_dataset(train_paths, val_paths)

    print("\nBuilding Faster R-CNN manifests...")
    build_frcnn_manifest(train_paths, FRCNN_MANIFEST_TRAIN)
    build_frcnn_manifest(val_paths, FRCNN_MANIFEST_VAL)

    print("\nDone. Classes:")
    for i, c in enumerate(CLASSES):
        print(f"  YOLO class {i} / FRCNN label {i+1}: {c}")


if __name__ == "__main__":
    main()
