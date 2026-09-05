# Training — full-image pipeline (current)

Trains each model on the 877 full images with real PASCAL VOC XML bounding
boxes. All 9 scripts share the same conventions:

- `OCCLUDED_DIR="occ"`, `MASK_DIR="occ_masks"`, `CLEAN_DIR="d"`, `ANNOT_DIR="e"`
- Reads `traintestsplit/data_split.json`
- Saves checkpoints with `{'model_state_dict', 'in_ch', 'arch', 'model_name'}`
  (Model G also saves `'discriminator_state_dict'`) — this format is what
  `evaluation/evaluate_all_models.py` and `detection/fullimage/*.py` expect.

Run all 9 sequentially with `train_all_9_models_fullimage.py`, or train any
one individually with its own script (identical logic, kept separate for
readability/standalone use).
