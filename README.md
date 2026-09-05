# Occlusion-Aware Road Sign Reconstruction & Detection Recovery

This repository contains the full pipeline for a project comparing **9 inpainting
architectures** on their ability to reconstruct occluded road signs, and measuring
how much that reconstruction actually **recovers object-detector performance**
(YOLOv5 and Faster R-CNN) on the repaired images.

## TL;DR result

**Model E (residual blocks + ASPP + attention-gated skip connections + composite
loss) is the best model overall** — it wins on SSIM, MS-SSIM, LPIPS, and on
post-inpainting detection recall at every high-occlusion level, across two
detector architectures and two different dataset formats (cropped patches and
full scenes). See [`docs/findings.md`](docs/findings.md) for the full writeup,
including two genuine failure-case studies (a collapsed GAN and a ViT's
patch-blocking artifacts).

## Project structure

```
├── training/
│   ├── fullimage/              <- CURRENT pipeline: trains on the 877 full
│   │   │                          images with real PASCAL VOC XML bounding
│   │   │                          boxes (not synthetic full-frame boxes)
│   │   ├── train_model_A_base_unet.py
│   │   ├── train_model_B_mask_unet.py
│   │   ├── train_model_C_roi_unet.py
│   │   ├── train_model_D_enhanced_unet.py
│   │   ├── train_model_E_advanced_unet.py      <- best model
│   │   ├── train_model_F_plain_autoencoder.py
│   │   ├── train_model_G_context_encoder_gan.py
│   │   ├── train_model_H_partial_conv_unet.py
│   │   ├── train_model_I_vit_inpainter.py
│   │   └── train_all_9_models_fullimage.py     <- all 9 in one script
│   └── legacy_cropped/          <- SUPERSEDED: earlier version trained on
│                                    1244 pre-cropped single-sign patches
│
├── evaluation/
│   ├── evaluate_all_models.py       <- PSNR/SSIM/MS-SSIM/LPIPS for all 9
│   │                                    models, full-image pipeline
│   └── evaluate_baseline_models.py  <- legacy cropped-patch version
│
├── detection/
│   ├── fullimage/                <- CURRENT: real multi-object, multi-class
│   │   ├── prepare_detector_dataset_fullimage.py
│   │   ├── train_yolov5_fullimage.py
│   │   ├── train_faster_rcnn_fullimage.py
│   │   ├── evaluate_detection_fullimage.py   <- IoU-based P/R/F1
│   │   └── diagnose_G_I.py                   <- failure-case diagnostic
│   └── legacy_cropped/           <- SUPERSEDED: synthetic full-frame boxes,
│                                    class-presence-only P/R (not IoU-based)
│
├── diagrams/
│   ├── drawio/                   <- editable diagrams (import into
│   │                                 diagrams.net or Lucidchart)
│   ├── rendered/                 <- PNG exports
│   └── model_icons/              <- one icon per model (A-I), for use in
│                                    diagrams/slides
│
└── docs/
    └── findings.md                <- full experimental writeup
```

## Pipeline overview

```
1. Data prep       clean full images (877) + PASCAL VOC XML annotations
                    -> synthetic occlusion generator (9 levels: 0-80%)
                    -> occluded images + occlusion masks

2. Model training   9 architecturally distinct models trained on the SAME
                    train/test split:
                      A: Base U-Net              (plain MSE, 3ch)
                      B: Mask U-Net              (plain MSE, 5ch)
                      C: ROI U-Net               (region-weighted MSE)
                      D: Enhanced U-Net          (+ SSIM + L1)
                      E: Advanced U-Net          (+ residual/ASPP/attention) *
                      F: Plain Autoencoder       (no skip connections)
                      G: Context-Encoder GAN     (adversarial)
                      H: Partial-Conv U-Net      (mask-aware convolutions)
                      I: ViT Inpainter           (pure transformer)

3. Model selection  evaluate_all_models.py ranks all 9 on PSNR/SSIM/
                    MS-SSIM/LPIPS -> Model E wins

4. Detector         train YOLOv5 + Faster R-CNN on the CLEAN images only
   training         (detectors never see occlusion during training)

5. Detection        For each model, each occlusion level: inpaint the
   recovery test     occluded image, then run both detectors on it.
                    Compare recall against the raw occluded image (no
                    inpainting) and the clean image (ceiling).
                    -> Model E recovers the most detection performance
                       at every high-occlusion level, on both detectors.
```

## Requirements

```bash
pip install torch torchvision opencv-python numpy pandas openpyxl matplotlib
pip install scikit-image pytorch-msssim lpips ultralytics
```

`ultralytics` is used for YOLOv5 training/inference (no repo clone needed).
`lpips` will download pretrained AlexNet/VGG weights on first use — needs
internet access once.

## Usage

### 1. Prepare your data

You need, sitting in your working directory:
```
d/                       clean full images
occ/                     occluded full images (synthetically generated)
occ_masks/               occlusion masks
e/                       PASCAL VOC XML annotations (one per clean image)
traintestsplit/data_split.json
```

`data_split.json` format:
```json
{
  "train": [{"image_id": "...", "category": "...", "occ_level": 40,
             "clean_path": "d/...", "occ_path": "occ/40/...",
             "mask_path": "occ_masks/40/..."}, ...],
  "test":  [...]
}
```

### 2. Train all 9 inpainting models

```bash
cd training/fullimage
python train_all_9_models_fullimage.py
# or train them individually, e.g.:
python train_model_E_advanced_unet.py
```

### 3. Evaluate reconstruction quality

```bash
cd evaluation
python evaluate_all_models.py
# -> outsept26/comparison_all_models.xlsx
```

### 4. Train the detectors (on clean images only)

```bash
cd detection/fullimage
python prepare_detector_dataset_fullimage.py
python train_yolov5_fullimage.py
python train_faster_rcnn_fullimage.py
```

### 5. Evaluate detection recovery

```bash
python evaluate_detection_fullimage.py
# -> detection_results_fullimage/detection_results_fullimage.xlsx
```

## Key findings (see docs/findings.md for details)

- **Model E wins on every perceptual/structural metric** (SSIM, MS-SSIM, LPIPS)
  and on detection recall at every occlusion level ≥30%, on both YOLOv5 and
  Faster R-CNN, on both the cropped-patch and full-image dataset formats.
- **Model G (GAN) suffers a genuine, severe reconstruction failure** — its
  output doesn't resemble a road sign at all (see the diagnostic images),
  consistent with catastrophic training instability rather than a code bug.
- **Model I (ViT) reconstructs the correct rough shape but with severe
  16x16 patch-blocking artifacts** that a detector trained on natural images
  can't recognize, despite being visually interpretable to a human — a
  distinct failure mode from Model G's.
- **Faster R-CNN degrades under occlusion far more severely than YOLOv5**,
  but also recovers more from inpainting — inpainting matters more for
  Faster R-CNN pipelines specifically.

## Notes on the two pipeline generations

This repo contains **two generations** of the same pipeline:

- **`fullimage/`** (current, correct): trained/evaluated on the 877 original
  images with real PASCAL VOC bounding boxes. Detection evaluation uses
  proper IoU-based matching (a real, if simplified, object-detection metric).
- **`legacy_cropped/`** (superseded): an earlier iteration that used 1244
  pre-cropped single-sign patches with *synthesized* full-frame bounding
  boxes (since the crops had no finer box to recover) and a simplified
  "does the right class appear anywhere in the image" detection metric.
  Kept for reference/reproducibility of earlier results, but the
  `fullimage/` pipeline is more rigorous and is the recommended one to use.
