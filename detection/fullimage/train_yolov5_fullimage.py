"""
train_yolov5_fullimage.py
============================
Trains YOLOv5 on yolo_dataset_fullimage/ (real multi-object, multi-class
labels from prepare_detector_dataset_fullimage.py), using the
`ultralytics` pip package -- no git/repo clone needed.

BEFORE RUNNING:
    pip install ultralytics
    python prepare_detector_dataset_fullimage.py

Usage:
    python train_yolov5_fullimage.py
"""

import os

DATA_YAML    = os.path.abspath("yolo_dataset_fullimage/data.yaml")
WEIGHTS_INIT = "yolov5su.pt"
IMG_SIZE     = 640      # full scenes -- use YOLO's standard size, not the 256 inpainting crop size
BATCH_SIZE   = 16
EPOCHS       = 100
PROJECT_DIR  = os.path.abspath("yolo_runs_fullimage")
RUN_NAME     = "sign_detector_fullimage"


def main():
    from ultralytics import YOLO

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found -- run prepare_detector_dataset_fullimage.py first.")

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
    print("Point evaluate_detection_fullimage.py's YOLOV5_WEIGHTS at this file.")


if __name__ == "__main__":
    main()
