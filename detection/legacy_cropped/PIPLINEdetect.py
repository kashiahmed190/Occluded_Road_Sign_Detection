"""
run_full_inpainting_detection_pipeline.py
============================================
ONE script that does everything:

  1. Runs all 9 trained inpainting models (A-I) on every TEST image,
     across all occlusion levels, and SAVES each inpainted image to disk:

        inpainting_detection_results/
        └── inpainted_images/
            ├── Model_A_Base_UNet/
            │   ├── 10/  <category>_<id>.png  ...
            │   ├── 20/  ...
            │   └── ... (one folder per occlusion level)
            ├── Model_B_Mask_UNet/
            ├── ...
            └── Model_I_ViT_Inpainter/

  2. Runs YOLOv5 + Faster R-CNN on:
       - the raw occluded image (pre-inpaint baseline, shared across models)
       - every model's inpainted output (post-inpaint)

  3. Saves everything to Excel/CSV in the SAME output folder:
        inpainting_detection_results/detection_results.xlsx     (per model x level)
        inpainting_detection_results/detection_results_summary.csv (mean per model)
        inpainting_detection_results/model_ranking.xlsx          (who wins, per detector)

This reuses the already-verified model loading / detector wrapper code
from evaluate_detection.py (same directory) rather than duplicating it --
make sure evaluate_detection.py is configured correctly (YOLOV5_WEIGHTS,
FRCNN_WEIGHTS, etc.) before running this.

Usage:
    python run_full_inpainting_detection_pipeline.py
"""

import os
import time
import numpy as np
import pandas as pd
import cv2
import torch

import Evalfinal as ed   # reuses YoloDetector, FrcnnDetector, model registry, etc.

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

OUTPUT_ROOT = "inpainting_detection_results"
IMAGES_DIR  = os.path.join(OUTPUT_ROOT, "inpainted_images")
SAVE_IMAGES = True   # set False to skip writing PNGs (much faster, metrics only)


def save_image(path, img_uint8_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2BGR))


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Loading detectors...")
    yolo = ed.YoloDetector(ed.YOLOV5_REPO_PATH, ed.YOLOV5_WEIGHTS,
                            ed.YOLOV5_CONF_THRES, ed.YOLOV5_IOU_THRES)
    frcnn = ed.FrcnnDetector(ed.FRCNN_WEIGHTS, ed.FRCNN_NUM_CLASSES, ed.FRCNN_CONF_THRES)

    print("Loading test split...")
    test_samples = ed.load_split(ed.SPLIT_PATH)

    # ── Standalone clean-dataset baseline ──────────────────────────────────
    # This is the TRUE reference point: raw clean images, straight into the
    # detectors, no occlusion and no inpainting model involved at all. Every
    # other number in this run (pre-inpaint on occluded images, post-inpaint
    # on any model's output) should be compared against THIS, not against
    # each other -- it's the ceiling nothing else can exceed except by noise.
    seen = {}
    for s in test_samples:
        key = s.get('image_id', s['clean_path'])
        seen[key] = s
    clean_samples = list(seen.values())

    print(f"\nRunning detection on the actual clean dataset ({len(clean_samples)} images, "
          f"no occlusion, no inpainting)...")
    acc_yolo_clean = ed.ConfusionAccumulator(ed.CLASSES)
    acc_frcnn_clean = ed.ConfusionAccumulator(ed.CLASSES)
    for s in clean_samples:
        clean_uint8 = (ed.load_rgb(s['clean_path'], ed.TARGET_SIZE) * 255).astype(np.uint8)
        cid = ed.class_id_for(s)
        acc_yolo_clean.update(cid, yolo.predict_classes(clean_uint8))
        acc_frcnn_clean.update(cid, frcnn.predict_classes(clean_uint8))

    clean_yolo_macro = acc_yolo_clean.macro()
    clean_frcnn_macro = acc_frcnn_clean.macro()
    print(f"  YOLOv5:      Precision={clean_yolo_macro['Precision_macro']:.3f}  "
          f"Recall={clean_yolo_macro['Recall_macro']:.3f}")
    print(f"  Faster R-CNN: Precision={clean_frcnn_macro['Precision_macro']:.3f}  "
          f"Recall={clean_frcnn_macro['Recall_macro']:.3f}")

    clean_baseline_rows = []
    for det_name, acc in [('YOLO', acc_yolo_clean), ('FRCNN', acc_frcnn_clean)]:
        for pc in acc.per_class_table():
            clean_baseline_rows.append({'Detector': det_name, **pc})
    clean_baseline_df = pd.DataFrame(clean_baseline_rows)

    # Synthesize the clean / 0% occlusion baseline (see comment history --
    # data_split.json only has rows for occ_level 10-80).
    level0_samples = []
    for s in seen.values():
        level0_samples.append({
            'image_id': s.get('image_id', os.path.splitext(os.path.basename(s['clean_path']))[0]),
            'category': s.get('category', 'unk'),
            'occ_level': 0, 'clean_path': s['clean_path'],
            'occ_path': s['clean_path'], 'mask_path': '',
        })
    test_samples = test_samples + level0_samples
    print(f"\n  Synthesized {len(level0_samples)} clean (0%) baseline samples "
          f"(same images, routed through each model to check for self-degradation)")

    level_groups = ed.group_by_level(test_samples)
    for lvl, s in level_groups.items():
        print(f"  Level {lvl:>3}% : {len(s)} test samples")

    registry, classes_reg = ed.build_model_registry()

    # ── pass 1: pre-inpaint (occluded image) detection, shared across all models ──
    print("\nRunning pre-inpaint detection on raw occluded images...")
    preinpaint = {}
    for lvl in ed.OCCLUSION_LEVELS:
        samples = level_groups.get(lvl, [])
        if not samples:
            continue
        acc_yolo = ed.ConfusionAccumulator(ed.CLASSES)
        acc_frcnn = ed.ConfusionAccumulator(ed.CLASSES)
        for s in samples:
            occ_uint8 = (ed.load_rgb(s['occ_path'], ed.TARGET_SIZE) * 255).astype(np.uint8)
            cid = ed.class_id_for(s)
            acc_yolo.update(cid, yolo.predict_classes(occ_uint8))
            acc_frcnn.update(cid, frcnn.predict_classes(occ_uint8))
        preinpaint[lvl] = {'yolo': acc_yolo, 'frcnn': acc_frcnn}
        my, mf = acc_yolo.macro(), acc_frcnn.macro()
        print(f"  Level {lvl:>3}%  YOLO P={my['Precision_macro']:.3f} R={my['Recall_macro']:.3f}  "
              f"FRCNN P={mf['Precision_macro']:.3f} R={mf['Recall_macro']:.3f}")

    # ── pass 2: per-model inpainting -> save images -> post-inpaint detection ──
    all_rows = []
    confusion_rows = []
    for cfg in registry:
        if not os.path.exists(cfg["ckpt"]):
            print(f"\nSkipping {cfg['name']} -- checkpoint not found at {cfg['ckpt']}")
            continue

        model_full_name = cfg["name"].replace(" ", "_").replace("(", "").replace(")", "")

        print(f"\n{'='*64}\n  {cfg['name']}\n{'='*64}")
        model = ed.load_inpaint_model(cfg, classes_reg)

        for lvl in ed.OCCLUSION_LEVELS:
            samples = level_groups.get(lvl, [])
            if not samples:
                continue

            acc_yolo = ed.ConfusionAccumulator(ed.CLASSES)
            acc_frcnn = ed.ConfusionAccumulator(ed.CLASSES)
            t0 = time.time()
            for i, s in enumerate(samples):
                x, occ_mask, _ = ed.build_input_tensor(s, ed.TARGET_SIZE)
                x, occ_mask = x.to(ed.DEVICE), occ_mask.to(ed.DEVICE)
                pred = ed.run_inpaint(model, cfg, x, occ_mask)
                pred_uint8 = ed.tensor_to_uint8(pred)

                if SAVE_IMAGES:
                    out_path = os.path.join(
                        IMAGES_DIR, model_full_name, str(lvl),
                        f"{s.get('category', 'unk')}_{s['image_id']}.png"
                        if 'image_id' in s else f"sample_{i}.png")
                    save_image(out_path, pred_uint8)

                cid = ed.class_id_for(s)
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

            for det_name, acc_post in [('YOLO', acc_yolo), ('FRCNN', acc_frcnn)]:
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
        print("\nNo models were evaluated -- check checkpoint paths in evaluate_detection.py's "
              "build_model_registry().")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        clean_baseline_df.to_csv(os.path.join(OUTPUT_ROOT, "clean_dataset_baseline.csv"), index=False)
        print(f"Saved clean-dataset baseline anyway -> "
              f"{os.path.join(OUTPUT_ROOT, 'clean_dataset_baseline.csv')}")
        return

    full_df = pd.DataFrame(all_rows)
    conf_df = pd.DataFrame(confusion_rows)
    detection_xlsx = os.path.join(OUTPUT_ROOT, "detection_results.xlsx")
    with pd.ExcelWriter(detection_xlsx, engine="openpyxl") as writer:
        clean_baseline_df.to_excel(writer, sheet_name="Clean Dataset Baseline", index=False)
        full_df.to_excel(writer, sheet_name="Detection Results", index=False)
        conf_df.to_excel(writer, sheet_name="Per-category P R F1", index=False)
    print(f"\nSaved -> {detection_xlsx} (3 sheets)")

    # ── per-model overview (mean across all levels) ──
    overview = (full_df.groupby('Model')[
        ['YOLO_Precision_pre', 'YOLO_Recall_pre', 'YOLO_Precision_post', 'YOLO_Recall_post',
         'YOLO_Recall_gain', 'FRCNN_Precision_pre', 'FRCNN_Recall_pre',
         'FRCNN_Precision_post', 'FRCNN_Recall_post', 'FRCNN_Recall_gain']
    ].mean().reset_index())
    summary_csv = os.path.join(OUTPUT_ROOT, "detection_results_summary.csv")
    overview.to_csv(summary_csv, index=False)

    # ── ranking: which model's inpainting helps each detector most ──
    yolo_rank = overview[['Model', 'YOLO_Recall_post', 'YOLO_Precision_post', 'YOLO_Recall_gain']].sort_values(
        'YOLO_Recall_post', ascending=False).reset_index(drop=True)
    yolo_rank.insert(0, 'Rank', yolo_rank.index + 1)

    frcnn_rank = overview[['Model', 'FRCNN_Recall_post', 'FRCNN_Precision_post', 'FRCNN_Recall_gain']].sort_values(
        'FRCNN_Recall_post', ascending=False).reset_index(drop=True)
    frcnn_rank.insert(0, 'Rank', frcnn_rank.index + 1)

    ranking_xlsx = os.path.join(OUTPUT_ROOT, "model_ranking.xlsx")
    with pd.ExcelWriter(ranking_xlsx, engine='openpyxl') as writer:
        overview.to_excel(writer, sheet_name='Overview (all models)', index=False)
        yolo_rank.to_excel(writer, sheet_name='Ranked for YOLOv5', index=False)
        frcnn_rank.to_excel(writer, sheet_name='Ranked for Faster R-CNN', index=False)
    print(f"Saved -> {ranking_xlsx}")

    best_yolo = yolo_rank.iloc[0]
    best_frcnn = frcnn_rank.iloc[0]

    print(f"\n{'='*64}")
    print("  RESULTS (macro precision/recall, mean across all occlusion levels)")
    print(f"{'='*64}")
    print(f"  Clean dataset (true ceiling, no occlusion/inpainting):")
    print(f"    YOLOv5:       Precision={clean_yolo_macro['Precision_macro']:.3f}  "
          f"Recall={clean_yolo_macro['Recall_macro']:.3f}")
    print(f"    Faster R-CNN: Precision={clean_frcnn_macro['Precision_macro']:.3f}  "
          f"Recall={clean_frcnn_macro['Recall_macro']:.3f}")
    print()
    print(overview.to_string(index=False))
    print(f"\nBest inpainting model for YOLOv5 post-inpaint recall:")
    print(f"  {best_yolo['Model']}  (recall={best_yolo['YOLO_Recall_post']:.3f}, "
          f"precision={best_yolo['YOLO_Precision_post']:.3f}, "
          f"gain=+{best_yolo['YOLO_Recall_gain']:.3f})")
    print(f"\nBest inpainting model for Faster R-CNN post-inpaint recall:")
    print(f"  {best_frcnn['Model']}  (recall={best_frcnn['FRCNN_Recall_post']:.3f}, "
          f"precision={best_frcnn['FRCNN_Precision_post']:.3f}, "
          f"gain=+{best_frcnn['FRCNN_Recall_gain']:.3f})")
    print(f"\nAll outputs saved under: {os.path.abspath(OUTPUT_ROOT)}/")
    if SAVE_IMAGES:
        print(f"  Inpainted images: {os.path.abspath(IMAGES_DIR)}/<Model>/<level>/*.png")
    print(f"  Per-level results: {detection_xlsx}")
    print(f"  Per-model summary: {summary_csv}")
    print(f"  Rankings:          {ranking_xlsx}")


if __name__ == "__main__":
    main()
