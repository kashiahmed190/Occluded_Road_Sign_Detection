"""
train_yolov5.py
=================
Trains YOLOv5 on the yolo_dataset/ produced by prepare_detector_dataset.py,
using the `ultralytics` pip package's Python API.

This avoids needing git or a cloned github.com/ultralytics/yolov5 repo --
useful on offline/restricted clusters (e.g. Singularity containers without
git) where you already have a working pip mirror.

BEFORE RUNNING:
    pip install ultralytics
    python prepare_detector_dataset.py      # if you haven't already

The very first run will still need to fetch pretrained YOLOv5 weights
(yolov5su.pt) once -- if that also fails in a fully offline environment,
see the FALLBACK note at the bottom of this file.

Usage:
    python train_yolov5.py
"""

import os

DATA_YAML    = os.path.abspath("yolo_dataset/data.yaml")
WEIGHTS_INIT = "yolov5su.pt"   # small YOLOv5 model, ultralytics-format weights
IMG_SIZE     = 256             # matches your crop resolution
BATCH_SIZE   = 16
EPOCHS       = 100
PROJECT_DIR  = os.path.abspath("yolo_runs")
RUN_NAME     = "sign_detector"


def main():
    from ultralytics import YOLO

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found -- run prepare_detector_dataset.py first.")

    model = YOLO(WEIGHTS_INIT)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    best_pt = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\nDone. Trained weights: {best_pt}")
    print("Point evaluate_detection.py's YOLOV5_WEIGHTS at this file.")
    print("Since this used the ultralytics package (not a cloned repo), also set")
    print("YOLOV5_REPO_PATH = None in evaluate_detection.py -- see the note there")
    print("about loading with the ultralytics API instead of torch.hub('local').")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════
# FALLBACK -- fully offline, no internet for ANY weight download
# ═══════════════════════════════════════════════════════════════
# If even yolov5su.pt can't be fetched, train from a random init instead:
#     model = YOLO("yolov5s.yaml")   # architecture only, no pretrained weights
# It will need more epochs to converge but requires zero downloads.
