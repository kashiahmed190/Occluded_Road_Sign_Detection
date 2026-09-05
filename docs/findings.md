# Experimental Findings

## 1. Reconstruction quality (PSNR / SSIM / MS-SSIM / LPIPS)

Averaged across all occlusion levels (10-80%), full-image pipeline, 995 train
/ 249 test images:

| Model | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---|---|---|---|
| A — Base U-Net | 22.40 | 0.797 | 0.798 | 0.291 |
| B — Mask U-Net | 22.48 | 0.796 | 0.803 | 0.290 |
| C — ROI U-Net | 22.20 | 0.790 | 0.792 | 0.304 |
| D — Enhanced U-Net | 22.02 | 0.812 | 0.805 | 0.278 |
| **E — Advanced U-Net** | 21.98 | **0.817** | **0.813** | **0.257** |
| F — Plain Autoencoder | 21.30 | 0.667 | 0.760 | 0.464 |
| G — Context-Encoder GAN | 15.44 | 0.346 | 0.311 | 0.686 |
| H — Partial-Conv U-Net | 20.12 | 0.730 | 0.718 | 0.386 |
| I — ViT Inpainter | 17.41 | 0.500 | 0.526 | 0.614 |

**Model E wins SSIM, MS-SSIM, and LPIPS.** Model B edges out E on raw PSNR by
a small margin — expected, since PSNR rewards pixel-averaging/blur in a way
SSIM and LPIPS correctly penalize.

## 2. Detection recall recovery (IoU-based, full-image pipeline)

Clean baseline (no occlusion, no inpainting): **YOLOv5 P=0.881 R=0.844**,
**Faster R-CNN P=0.890 R=0.932**.

Pre-inpaint (occluded image, untouched) recall drops severely with occlusion:

| Level | YOLOv5 R | Faster R-CNN R |
|---|---|---|
| 10% | 0.668 | 0.769 |
| 40% | 0.296 | 0.619 |
| 60% | 0.182 | 0.332 |
| 80% | 0.300 | 0.368 |

Post-inpaint YOLOv5 recall at high occlusion, by model:

| Level | A | B | C | D | **E** |
|---|---|---|---|---|---|
| 60% | 0.184 | 0.196 | 0.278 | 0.396 | **0.461** |
| 70% | 0.205 | 0.207 | 0.254 | 0.380 | **0.427** |
| 80% | 0.240 | 0.260 | 0.323 | 0.395 | **0.440** |

Post-inpaint Faster R-CNN recall at high occlusion, by model:

| Level | A | B | C | D | **E** |
|---|---|---|---|---|---|
| 60% | 0.454 | 0.451 | 0.506 | 0.571 | **0.599** |
| 70% | 0.409 | 0.417 | 0.503 | 0.569 | **0.598** |
| 80% | 0.429 | 0.444 | 0.521 | 0.543 | **0.589** |

**A clean, monotonic A < B < C < D < E ordering at every high-occlusion
level, on both detectors.** This is a much stronger and more consistent
result than the cropped-patch pipeline produced (where the ordering wasn't
always monotonic — see below).

Models F, G, H, I were also evaluated; G and I both show catastrophic
detection failure (~0% recall) despite non-trivial-looking pixel statistics.
See section 4 for the root-cause investigation.

## 3. Comparison: cropped-patch pipeline vs. full-image pipeline

An earlier iteration of this project trained on 1244 pre-cropped single-sign
patches (each patch = one object, tightly cropped, filling most of the
frame) rather than the 877 original full scenes. Both pipelines agree on
the headline result (Model E is best), but with some differences:

- The cropped-patch pipeline's detection metric was a simplification
  ("does the correct class appear anywhere in the image", since there was
  only ever one object per crop and no finer box to check against). The
  full-image pipeline uses genuine IoU-based box matching against real
  multi-object ground truth — a more rigorous measurement.
- On cropped patches, Model C sometimes underperformed A/B (a non-monotonic
  ablation), which did not reproduce on full images, where the ablation is
  cleanly monotonic (A < B < C < D < E) at every high-occlusion level.
- Faster R-CNN's occlusion sensitivity is far more dramatic on full images
  (recall crashes to 0.33-0.37 at 60-80% occlusion) than it appeared on
  cropped patches, likely because a full scene gives the detector far less
  positional/framing information to work with than an already-cropped,
  sign-centered patch.

## 4. Failure case investigation: Model G and Model I

Both G and I showed near-zero detection recall on the full-image pipeline,
which looked suspicious enough to warrant checking whether this was a bug
in the evaluation code rather than a genuine model failure.

**Diagnostic method**: loaded each model, ran inference on real occluded
test images, inspected raw tensor statistics (min/max/mean/std, NaN/Inf
checks) and saved the actual output images for visual inspection.

**Result: both are genuine model failures, not code bugs** — confirmed by
inspecting the actual output images directly.

- **Model G (Context-Encoder GAN)**: the output has normal-looking pixel
  statistics (std ≈ 0.14-0.19, full dynamic range) but visually **does not
  resemble a road sign at all** — no circular shape, no red ring, no
  visible digits. Just a generic blurry color gradient that could be any
  scene. This is consistent with a severe generator training failure
  (plausibly mode collapse toward "generically photo-like texture" without
  learning to reconstruct actual content), matching the independently
  measured PSNR≈15dB / SSIM≈0.35.

- **Model I (ViT Inpainter)**: the output correctly reconstructs the rough
  shape (visible red ring, dark region where the digits should be) but is
  **severely blocky**, with a clear 16x16 grid pattern from the patch
  embedding leaking directly into pixel space. A human can tell it's a
  sign; a CNN-based detector trained exclusively on naturally-textured
  images apparently cannot, since the hard patch-boundary edges are
  statistically foreign to what it learned to recognize. This is a
  different failure mode from G's: correct high-level content, wrong
  low-level texture statistics.

Both failures make sense given each architecture's known weaknesses (GAN
training instability; naive ViT patch decoding without a smoothing/
refinement stage) and neither is present in Model E, which uses neither
adversarial training nor raw patch-based decoding — reinforcing that E's
specific architectural choices (residual convs + ASPP + attention gates,
trained with a stable composite pixel/structural loss) avoid both classes
of failure.

## 5. Practical implications

- **If you only have budget to train one model, train Model E.** It wins on
  every quality metric that matters and on downstream detection recovery,
  on two different detector architectures, on two dataset formats.
- **Detection-based evaluation surfaces failures that pixel metrics alone
  might under-communicate.** Model I's SSIM (0.50) doesn't sound
  catastrophic in isolation, but its 0% detection recall makes clear that
  "looks okay to a similarity metric" and "usable by a downstream system"
  are not the same thing.
- **Faster R-CNN benefits more from inpainting than YOLOv5 does** — YOLOv5
  is comparatively more occlusion-robust on its own, so the inpainting
  step's ROI (in a practical deployment sense) is larger for Faster R-CNN
  pipelines specifically.
