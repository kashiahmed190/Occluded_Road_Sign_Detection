import os
import json
import csv
import time
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================================================
# MODEL A — Base U-Net
#   Input  : 3 channels (RGB only -- no mask channels)
#   Loss   : plain MSE only
#   Purpose: weakest baseline -- no mask input, no region
#            weighting, no architectural improvements.
#            Isolates how much EVERYTHING else in B-E is worth.
# =========================================================

OCCLUDED_DIR    = "occ"
MASK_DIR        = "occ_masks"
CLEAN_DIR       = "d"
ANNOT_DIR       = "e"

BASE_SPLIT_PATH = os.path.join("traintestsplit", "data_split.json")
SAVE_DIR        = "inpainting_model"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_SIZE  = (256, 256)
BATCH_SIZE   = 8
EPOCHS       = 60
LR           = 1e-4
NUM_WORKERS  = 2
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 12
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH   = os.path.join(SAVE_DIR, "best_inpainting.pth")
FINAL_MODEL_PATH  = os.path.join(SAVE_DIR, "final_inpainting.pth")
LOSS_PLOT_PATH    = os.path.join(SAVE_DIR, "loss_curve.png")
HISTORY_JSON_PATH = os.path.join(SAVE_DIR, "history.json")
HISTORY_CSV_PATH  = os.path.join(SAVE_DIR, "epoch_log.csv")
CONFIG_JSON_PATH  = os.path.join(SAVE_DIR, "training_config.json")
SUMMARY_JSON_PATH = os.path.join(SAVE_DIR, "training_summary.json")
SPLIT_COPY_PATH   = os.path.join(SAVE_DIR, "split_used.json")

print("Device:", DEVICE)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def load_existing_split(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        split = json.load(f)
    train_samples = split["train"]
    val_samples   = split.get("test", split.get("val", []))
    print(f"Split loaded: train={len(train_samples)}  val={len(val_samples)}\n")
    return train_samples, val_samples, split


def parse_annotation(xml_path, original_hw, target_hw):
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        H0, W0 = original_hw
        Ht, Wt = target_hw
        sx = Wt / float(W0)
        sy = Ht / float(H0)
        bboxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = int(float(bb.find("xmin").text) * sx)
            ymin = int(float(bb.find("ymin").text) * sy)
            xmax = int(float(bb.find("xmax").text) * sx)
            ymax = int(float(bb.find("ymax").text) * sy)
            xmin = max(0, min(Wt-1, xmin))
            xmax = max(0, min(Wt,   xmax))
            ymin = max(0, min(Ht-1, ymin))
            ymax = max(0, min(Ht,   ymax))
            if xmax > xmin and ymax > ymin:
                bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")
        return []

def make_bbox_mask(h, w, bboxes):
    mask = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 1.0
    return mask


class BaseUNetDataset(Dataset):
    """3-channel input -- RGB only. occ_mask/bbox_mask still loaded/returned
    (unused as model input) so the val loop can still report masked metrics
    consistently with the other models, if you choose to."""
    def __init__(self, samples, annot_dir, mask_dir,
                 target_size=(256, 256), augment=False):
        self.samples     = samples
        self.annot_dir   = annot_dir
        self.mask_dir    = mask_dir
        self.target_size = target_size
        self.augment     = augment

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read: {path}")
        H0, W0 = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0, (H0, W0)

    def _load_occ_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            Ht, Wt = self.target_size[1], self.target_size[0]
            return np.zeros((Ht, Wt), dtype=np.float32)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)

    def _augment(self, occ, clean, occ_mask, bbox_mask):
        if np.random.rand() > 0.5:
            occ       = occ      [:, ::-1, :].copy()
            clean     = clean    [:, ::-1, :].copy()
            occ_mask  = occ_mask [:, ::-1   ].copy()
            bbox_mask = bbox_mask[:, ::-1   ].copy()
        if np.random.rand() > 0.8:
            occ       = occ      [::-1, :, :].copy()
            clean     = clean    [::-1, :, :].copy()
            occ_mask  = occ_mask [::-1, :   ].copy()
            bbox_mask = bbox_mask[::-1, :   ].copy()
        factor = np.random.uniform(0.8, 1.2)
        occ    = np.clip(occ   * factor, 0.0, 1.0)
        clean  = np.clip(clean * factor, 0.0, 1.0)
        return occ, clean, occ_mask, bbox_mask

    def __getitem__(self, idx):
        s = self.samples[idx]
        occ_img,   _           = self._load_rgb(s['occ_path'])
        clean_img, original_hw = self._load_rgb(s['clean_path'])
        occ_mask_path = s.get('mask_path', '')
        occ_mask      = self._load_occ_mask(occ_mask_path)

        Ht, Wt   = self.target_size[1], self.target_size[0]
        base     = os.path.splitext(os.path.basename(s['clean_path']))[0]
        xml_path = os.path.join(self.annot_dir, base + '.xml')
        bboxes   = parse_annotation(xml_path, original_hw, (Ht, Wt))
        bbox_mask = (np.ones((Ht, Wt), dtype=np.float32) if len(bboxes) == 0
                     else make_bbox_mask(Ht, Wt, bboxes))

        if self.augment:
            occ_img, clean_img, occ_mask, bbox_mask = \
                self._augment(occ_img, clean_img, occ_mask, bbox_mask)

        occ_img   = np.clip(occ_img,   0.0, 1.0)
        clean_img = np.clip(clean_img, 0.0, 1.0)
        occ_mask  = np.clip(occ_mask,  0.0, 1.0)
        bbox_mask = np.clip(bbox_mask, 0.0, 1.0)

        x = occ_img.astype(np.float32)          # <-- 3ch only, KEY DIFFERENCE
        y = clean_img.astype(np.float32)

        return (
            torch.from_numpy(x.transpose(2, 0, 1)).float(),
            torch.from_numpy(y.transpose(2, 0, 1)).float(),
            torch.from_numpy(occ_mask).unsqueeze(0).float(),
            torch.from_numpy(bbox_mask).unsqueeze(0).float(),
            os.path.basename(s['clean_path'])
        )


def collate_fn(batch):
    xs, ys, oms, bms, fns = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(oms), torch.stack(bms), list(fns)


def mse_loss_only(y_true, y_pred):
    return F.mse_loss(y_pred.float(), y_true.float())


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.drop(x)

class BaseUNet(nn.Module):
    """3ch input, same depth/channels as every other U-Net variant here."""
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.c1 = ConvBlock(in_ch, 64,   0.1); self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(64,   128,   0.1); self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(128,  256,   0.2); self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(256,  512,   0.2); self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(512,  1024,  0.3)
        self.u6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c6 = ConvBlock(1024, 512, 0.2)
        self.u7 = nn.ConvTranspose2d(512,  256, 2, stride=2)
        self.c7 = ConvBlock(512,  256, 0.2)
        self.u8 = nn.ConvTranspose2d(256,  128, 2, stride=2)
        self.c8 = ConvBlock(256,  128, 0.1)
        self.u9 = nn.ConvTranspose2d(128,   64, 2, stride=2)
        self.c9 = ConvBlock(128,   64, 0.1)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.c1(x);  p1 = self.p1(c1)
        c2 = self.c2(p1); p2 = self.p2(c2)
        c3 = self.c3(p2); p3 = self.p3(c3)
        c4 = self.c4(p3); p4 = self.p4(c4)
        c5 = self.c5(p4)
        c6 = self.c6(torch.cat([self.u6(c5), c4], dim=1))
        c7 = self.c7(torch.cat([self.u7(c6), c3], dim=1))
        c8 = self.c8(torch.cat([self.u8(c7), c2], dim=1))
        c9 = self.c9(torch.cat([self.u9(c8), c1], dim=1))
        return torch.sigmoid(self.out(c9))


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_history_csv(history, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss'])
        for i, (tr, va) in enumerate(zip(history['train'], history['val']), 1):
            w.writerow([i, float(tr), float(va)])

def save_loss_curve(history, path):
    plt.figure(figsize=(8, 4))
    plt.plot(history['train'], label='train')
    plt.plot(history['val'],   label='val')
    plt.title('Model A — Base U-Net (3ch input, MSE loss only)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def train():
    start_time = time.time()
    print("=" * 60)
    print("  MODEL A — Base U-Net")
    print("  Input : 3ch (RGB only)")
    print("  Loss  : plain MSE")
    print("=" * 60)

    train_samples, val_samples, split_obj = load_existing_split(BASE_SPLIT_PATH)
    shutil.copy2(BASE_SPLIT_PATH, SPLIT_COPY_PATH)

    train_ds = BaseUNetDataset(train_samples, ANNOT_DIR, MASK_DIR, TARGET_SIZE, augment=True)
    val_ds   = BaseUNetDataset(val_samples,   ANNOT_DIR, MASK_DIR, TARGET_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=False)

    model     = BaseUNet(in_ch=3, out_ch=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train': [], 'val': []}

    save_json(CONFIG_JSON_PATH, {
        'model': 'Model A — Base U-Net', 'in_channels': 3,
        'loss': 'MSE only', 'BATCH_SIZE': BATCH_SIZE, 'EPOCHS': EPOCHS,
        'LR': LR, 'EARLY_STOP': EARLY_STOP, 'TARGET_SIZE': list(TARGET_SIZE),
        'DEVICE': str(DEVICE), 'n_train': len(train_samples), 'n_val': len(val_samples),
    })

    for epoch in range(EPOCHS):
        model.train()
        train_loss, nan_batches = 0.0, 0
        for x, y, _, _, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse_loss_only(y, pred)
            if not torch.isfinite(loss):
                nan_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        n_valid = max(1, len(train_loader) - nan_batches)
        train_loss /= n_valid
        if nan_batches > 0:
            print(f"  [WARN] {nan_batches} NaN batches skipped")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _, _, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = mse_loss_only(y, pred)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history['train'].append(float(train_loss))
        history['val'].append(float(val_loss))
        save_json(HISTORY_JSON_PATH, history)
        save_history_csv(history, HISTORY_CSV_PATH)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_epoch, patience_counter = float(val_loss), epoch + 1, 0
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val': best_val, 'history': history,
                'in_ch': 3, 'arch': 'unet', 'model_name': 'Model A — Base U-Net',
            }, BEST_MODEL_PATH)
            print(f"  Saved best model -> {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    torch.save({
        'model_state_dict': model.state_dict(), 'history': history,
        'in_ch': 3, 'arch': 'unet', 'model_name': 'Model A — Base U-Net',
    }, FINAL_MODEL_PATH)

    save_loss_curve(history, LOSS_PLOT_PATH)
    end_time = time.time()

    save_json(SUMMARY_JSON_PATH, {
        'model': 'Model A — Base U-Net', 'best_val_loss': best_val,
        'best_epoch': best_epoch, 'epochs_completed': len(history['train']),
        'duration_seconds': end_time - start_time,
        'best_model_path': BEST_MODEL_PATH, 'final_model_path': FINAL_MODEL_PATH,
    })

    print(f"\nTraining complete — Model A")
    print(f"  Best val loss : {best_val:.6f} at epoch {best_epoch}")
    print(f"  Best model    : {BEST_MODEL_PATH}")
    print(f"  Final model   : {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    train()
