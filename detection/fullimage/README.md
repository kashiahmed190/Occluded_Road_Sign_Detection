# Detection — full-image pipeline (current)

Run in this order:

1. `prepare_detector_dataset_fullimage.py` — extracts real multi-object,
   multi-class boxes from the XML annotations; builds a YOLOv5-format
   dataset and a Faster R-CNN manifest. Detectors are trained on **clean
   images only**.
2. `train_yolov5_fullimage.py` — uses the `ultralytics` pip package (no
   repo clone needed).
3. `train_faster_rcnn_fullimage.py` — standard torchvision Faster R-CNN
   fine-tuning on the real multi-object targets.
4. `evaluate_detection_fullimage.py` — the main evaluation: runs all 9
   inpainting models at all 9 occlusion levels, computes real IoU-based
   (threshold 0.5) precision/recall/F1 per class, and compares pre- vs.
   post-inpainting detection recall.
5. `diagnose_G_I.py` — saves actual output images + tensor stats for a
   given model, useful whenever a model's numbers look suspicious (this
   is how the Model G / Model I failure cases in `docs/findings.md` were
   confirmed as genuine rather than an evaluation bug).
