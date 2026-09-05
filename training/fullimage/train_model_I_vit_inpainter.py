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
# MODEL I — ViT Inpainter
#   Input  : 5 channels (RGB + occ_mask + bbox_mask)
#   Arch   : patch embedding + transformer encoder + linear
#            un-patchify -- NO convolutions, no encoder-decoder
#            hierarchy, no skip connections at all
#   Loss   : region-weighted MSE (real bbox_mask) + L1 (hidden)
#   Purpose: tests whether pure global attention beats local
#            convolutional inductive bias -- the architectural
#            opposite extreme from Model E
# =========================================================

OCCLUDED_DIR    = "occ"
MASK_DIR        = "occ_masks"
CLEAN_DIR       = "d"
ANNOT_DIR       = "e"

BASE_SPLIT_PATH = os.path.join("traintestsplit", "data_split.json")
SAVE_DIR        = "model_I_vit_inpainter"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_SIZE  = (256, 256)
BATCH_SIZE   = 8
EPOCHS       = 60
LR           = 1e-4
NUM_WORKERS  = 2
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 12
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_HIDDEN  = 1.0
W_VISIBLE = 0.4
W_BG      = 0.1

BEST_MODEL_PATH   = os.path.join(SAVE_DIR, "best_model_I.pth")
FINAL_MODEL_PATH  = os.path.join(SAVE_DIR, "final_model_I.pth")
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


class ViTDataset(Dataset):
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

        x = np.concatenate([occ_img, occ_mask[..., None], bbox_mask[..., None]],
                           axis=-1).astype(np.float32)
        y = clean_img.astype(np.float32)

        return (
            torch.from_numpy(x.transpose(2, 0, 1)).float(),
            torch.from_numpy(y.transpose(2, 0, 1)).float(),
            torch.from_numpy(occ_mask).unsqueeze(0).float(),
            torch.from_numpy(bbox_mask).unsqueeze(0).float(),
            os.path.basename(s['clean_path']),
        )


def collate_fn(batch):
    xs, ys, oms, bms, fns = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(oms), torch.stack(bms), list(fns)


class ViTInpainter(nn.Module):
    """Pure transformer -- patch embed, transformer encoder, linear
    un-patchify. No convolutional inductive bias anywhere."""
    def __init__(self, in_ch=5, out_ch=3, img_size=256, patch=16,
                 embed_dim=384, depth=6, n_heads=6, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        self.patch     = patch
        self.grid      = img_size // patch
        self.n_patches = self.grid * self.grid
        self.out_ch    = out_ch

        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch * patch * out_ch)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        pix = self.head(tokens)
        pix = pix.view(B, self.grid, self.grid, self.patch, self.patch, self.out_ch)
        pix = pix.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = pix.view(B, self.out_ch, self.grid * self.patch, self.grid * self.patch)
        return torch.sigmoid(img)


def region_weights(occ_mask, bbox_mask):
    occ  = occ_mask.float()
    bbox = bbox_mask.float()
    return (W_HIDDEN * bbox * occ + W_VISIBLE * bbox * (1.0 - occ) + W_BG * (1.0 - bbox))

def vit_loss(pred, gt, occ_mask, bbox_mask):
    """Model I -- region-weighted MSE + L1 on the hidden region (same
    recipe as Model D, so any gap vs Model E is attributable to the
    architecture, not the loss)."""
    w = region_weights(occ_mask, bbox_mask).expand_as(pred)
    mse = (w * (pred - gt) ** 2).mean()
    hole = (occ_mask * bbox_mask).expand_as(pred)
    n = hole.sum().clamp(min=1)
    l1 = ((pred - gt).abs() * hole).sum() / n
    return 0.7 * mse + 0.3 * l1


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
    plt.title('Model I — ViT Inpainter (region MSE + L1)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def train():
    start_time = time.time()
    print("=" * 60)
    print("  MODEL I — ViT Inpainter")
    print("  Input : 5ch (RGB + occ_mask + bbox_mask)")
    print("  Loss  : region-weighted MSE + L1 (hidden region)")
    print("=" * 60)

    train_samples, val_samples, split_obj = load_existing_split(BASE_SPLIT_PATH)
    shutil.copy2(BASE_SPLIT_PATH, SPLIT_COPY_PATH)

    train_ds = ViTDataset(train_samples, ANNOT_DIR, MASK_DIR, TARGET_SIZE, augment=True)
    val_ds   = ViTDataset(val_samples,   ANNOT_DIR, MASK_DIR, TARGET_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn, drop_last=False)

    model     = ViTInpainter(in_ch=5, out_ch=3, img_size=TARGET_SIZE[0]).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    save_json(CONFIG_JSON_PATH, {
        'model': 'Model I — ViT Inpainter', 'in_channels': 5,
        'loss': '0.7*region-MSE(real bbox) + 0.3*L1(hidden)', 'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS, 'LR': LR, 'EARLY_STOP': EARLY_STOP,
        'TARGET_SIZE': list(TARGET_SIZE), 'DEVICE': str(DEVICE),
        'n_train': len(train_samples), 'n_val': len(val_samples), 'n_params': n_params,
    })

    best_val, best_epoch, patience_counter = float('inf'), 0, 0
    history = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, nan_batches = 0.0, 0
        for x, y, occ_mask, bbox_mask, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            occ_mask, bbox_mask = occ_mask.to(DEVICE), bbox_mask.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = vit_loss(pred, y, occ_mask, bbox_mask)

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
            for x, y, occ_mask, bbox_mask, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                occ_mask, bbox_mask = occ_mask.to(DEVICE), bbox_mask.to(DEVICE)
                pred = model(x)
                loss = vit_loss(pred, y, occ_mask, bbox_mask)
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
                'in_ch': 5, 'arch': 'vit', 'model_name': 'Model I — ViT Inpainter',
            }, BEST_MODEL_PATH)
            print(f"  Saved best model -> {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    torch.save({
        'model_state_dict': model.state_dict(), 'history': history,
        'in_ch': 5, 'arch': 'vit', 'model_name': 'Model I — ViT Inpainter',
    }, FINAL_MODEL_PATH)

    save_loss_curve(history, LOSS_PLOT_PATH)
    end_time = time.time()

    save_json(SUMMARY_JSON_PATH, {
        'model': 'Model I — ViT Inpainter', 'best_val_loss': best_val,
        'best_epoch': best_epoch, 'epochs_completed': len(history['train']),
        'duration_seconds': end_time - start_time,
        'best_model_path': BEST_MODEL_PATH, 'final_model_path': FINAL_MODEL_PATH,
        'n_params': n_params,
    })

    print(f"\nTraining complete — Model I")
    print(f"  Best val loss : {best_val:.6f} at epoch {best_epoch}")
    print(f"  Best model    : {BEST_MODEL_PATH}")
    print(f"  Final model   : {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    train()
