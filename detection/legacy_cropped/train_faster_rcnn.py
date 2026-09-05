"""
train_faster_rcnn.py
======================
Trains a torchvision Faster R-CNN (ResNet-50 FPN) on the clean, per-
category sign crops (same images used for the U-Net models), using the
manifests produced by prepare_detector_dataset.py.

Run prepare_detector_dataset.py FIRST.

Output checkpoint is saved in the exact format evaluate_detection.py's
FrcnnDetector expects: a plain state_dict at FRCNN_WEIGHTS.

Usage:
    python train_faster_rcnn.py
"""

import os, json, time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

TRAIN_MANIFEST = "frcnn_manifest_train.json"
VAL_MANIFEST   = "frcnn_manifest_val.json"
CLASSES        = ["crosswalk", "speedlimit", "stop", "trafficlight"]
NUM_CLASSES    = len(CLASSES) + 1   # +1 for background (torchvision convention)

BATCH_SIZE   = 4
EPOCHS       = 20
LR           = 5e-3
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "faster_rcnn_sign_detector"
CKPT_PATH = os.path.join(SAVE_DIR, "best_frcnn.pth")

print(f"Device : {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class SignDetectionDataset(Dataset):
    def __init__(self, manifest_path, augment=False):
        with open(manifest_path) as f:
            self.rows = json.load(f)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row['clean_path'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment and np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
            h, w = img.shape[:2]
            x1, y1, x2, y2 = row['bbox']
            x1, x2 = w - x2, w - x1
            box = [x1, y1, x2, y2]
        else:
            box = row['bbox']

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        boxes = torch.tensor([box], dtype=torch.float32)
        labels = torch.tensor([row['label_id']], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((1,), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # COCO-pretrained backbone
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ═══════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOPS
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_loss(model, loader):
    """Faster R-CNN only returns losses in .train() mode, so we keep the
    model in train mode for this pass but disable gradient computation --
    this gives a comparable 'validation loss' for early stopping/checkpointing
    without actually updating any weights."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_ds = SignDetectionDataset(TRAIN_MANIFEST, augment=True)
    val_ds   = SignDetectionDataset(VAL_MANIFEST, augment=False)
    print(f"Train images: {len(train_ds)}   Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = build_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate_loss(model, val_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1:03d}/{EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"({time.time()-t0:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  Saved best checkpoint -> {CKPT_PATH} (val_loss={best_val:.4f})")

    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best checkpoint: {CKPT_PATH}")
    print(f"NUM_CLASSES used: {NUM_CLASSES} (background + {len(CLASSES)} sign classes)")
    print("Point evaluate_detection.py's FRCNN_WEIGHTS at this checkpoint and set "
          f"FRCNN_NUM_CLASSES = {NUM_CLASSES}.")


if __name__ == "__main__":
    main()
