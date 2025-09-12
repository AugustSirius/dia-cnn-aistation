#!/usr/bin/env python3
# ------------------------------------------------------------
# CNN-based peak-group classifier with automatic multi-GPU support
# 
# Just run: python train_peak_cnn.py
# It will automatically detect and use all available GPUs
# ------------------------------------------------------------
"""
Requirements
------------
pip install torch>=2.0 tqdm psutil
"""
from __future__ import annotations
import os, pickle, random, time, math, json
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from tqdm import tqdm, trange
import psutil

# --------------------------------------------------------------------------- #
#                            Configuration section                             #
# --------------------------------------------------------------------------- #

DATA_FOLDERS = {
    "decoy_base_reverse0"   : "/guotiannan/train_data_20250830/decoy_base_reverse0",
    "decoy_base_rt_shift"   : "/guotiannan/train_data_20250830/decoy_base_rt_shift",
    "diann20_target"        : "/guotiannan/train_data_20250830/diann20_target",
    "diann20_target_rt_shift":"/guotiannan/train_data_20250830/diann20_target_rt_shift",
}
LABELS = {                      # folder → class label
    "decoy_base_reverse0"    : 0,
    "decoy_base_rt_shift"    : 0,
    "diann20_target"         : 1,
    "diann20_target_rt_shift": 1,
}

EPOCHS          = 50
BATCH_SIZE      = 1024                      # per GPU batch size
NUM_WORKERS     = 8
LR              = 2e-4
WEIGHT_DECAY    = 1e-5
LOSS_FN         = "bce"                     # "bce" | "focal"
FOCAL_GAMMA     = 2.0
MODEL_SIZE      = "medium"                  # see presets below
VAL_SPLIT       = 0.02                      # 2 % validation
FILE_CACHE_K    = 32                        # keep ≤K files in RAM
SEED            = 42
PRINT_EVERY     = 200                       # batches

# --------------------------------------------------------------------------- #
#                          Auto GPU Detection                                 #
# --------------------------------------------------------------------------- #
def setup_device():
    """Automatically detect and setup GPUs"""
    if not torch.cuda.is_available():
        print("No CUDA devices found. Using CPU.")
        return torch.device("cpu"), 0
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s) available:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Use first GPU as primary device
    device = torch.device("cuda:0")
    return device, num_gpus

device, NUM_GPUS = setup_device()

# Adjust batch size for multiple GPUs
if NUM_GPUS > 1:
    TOTAL_BATCH_SIZE = BATCH_SIZE * NUM_GPUS
    print(f"Using DataParallel with {NUM_GPUS} GPUs")
    print(f"Effective batch size: {TOTAL_BATCH_SIZE} ({BATCH_SIZE} per GPU)")
else:
    TOTAL_BATCH_SIZE = BATCH_SIZE

# --------------------------------------------------------------------------- #
#                          Dataset class definition                           #
# --------------------------------------------------------------------------- #
class Dataset:
    """Original Dataset class used when creating the pickle files"""
    def __init__(self):
        self.rsm = None
        self.frag_info = None
        self.feat = None
        self.label = None
        self.file = None
        self.precursor_id = None

    def __getitem__(self, idx):
        return {
            "rsm": self.rsm[idx],
            "frag_info": self.frag_info[idx],
            "feat": self.feat[idx],
            "label": self.label[idx],
            "file": self.file[idx],
            "precursor_id": self.precursor_id[idx],
        }

    def __len__(self):
        return len(self.precursor_id) if self.precursor_id is not None else 0

    def fit_scale(self):
        pass

# --------------------------------------------------------------------------- #
#                       Model-complexity presets                               #
# --------------------------------------------------------------------------- #
PRESET = {
    "small" : dict(ch=[16,32,64,128],  fc=[128,64],  drop=[0.3,0.2]),
    "medium": dict(ch=[32,64,128,256], fc=[256,128], drop=[0.5,0.3]),
    "large" : dict(ch=[64,128,256,512],fc=[512,256], drop=[0.5,0.3]),
    "xlarge": dict(ch=[128,256,512,1024],fc=[1024,512],drop=[0.5,0.4]),
}[MODEL_SIZE]

# --------------------------------------------------------------------------- #
#                          Reproducibility                                     #
# --------------------------------------------------------------------------- #
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #
#                          Dataset indexing                                    #
# --------------------------------------------------------------------------- #

INDEX_PATH = Path(f"index_{len(DATA_FOLDERS)}folders.json")

def scan_dataset() -> dict:
    """
    Build or load an index:
    { "files": [ { "path": str, "n": int, "label": 0/1 }, ... ],
      "total" : int }
    """
    if INDEX_PATH.exists():
        print(f"Loading index {INDEX_PATH}")
        return json.loads(INDEX_PATH.read_text())

    print("Scanning folders and counting samples ...")

    files = []
    for name, folder in DATA_FOLDERS.items():
        for pkl_path in sorted(Path(folder).glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                ds = pickle.load(f)
                n = len(ds.rsm)                   # number of samples in file
            files.append(dict(path=str(pkl_path),
                              n=n,
                              label=LABELS[name]))
    idx = {"files": files,
           "total": int(sum(f["n"] for f in files))}
    INDEX_PATH.write_text(json.dumps(idx))
    print(f"Index saved to {INDEX_PATH}")
    return idx

INDEX = scan_dataset()

# --------------------------------------------------------------------------- #
#                             Dataset class                                   #
# --------------------------------------------------------------------------- #
class PeakDataset(td.Dataset):
    """
    Memory-efficient: keeps ≤FILE_CACHE_K .pkl files in RAM (LRU).
    Provides random access via a global (file_idx, local_idx) mapping.
    """
    def __init__(self, indices: np.ndarray):
        super().__init__()
        self.mapping = indices                                    # (N,2)
        self.file_meta = INDEX["files"]
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self): 
        return len(self.mapping)

    def _load_file(self, file_idx: int):
        meta = self.file_meta[file_idx]
        if file_idx in self.cache:
            self.cache.move_to_end(file_idx)
            return self.cache[file_idx]

        with open(meta["path"], "rb") as f:
            ds = pickle.load(f)
            # rsm (N,72, 8,16) → mean first 5 frag dims → (N,72,16)
            x = np.asarray(ds.rsm)[...,:5,:].mean(2).astype(np.float32)
            y = np.full(len(x), meta["label"], np.float32)

        self.cache[file_idx] = (x, y)
        self.cache.move_to_end(file_idx)
        # LRU eviction
        while len(self.cache) > FILE_CACHE_K:
            self.cache.popitem(last=False)
        return x, y

    def __getitem__(self, idx: int):
        file_idx, local_idx = self.mapping[idx]
        x, y = self._load_file(file_idx)
        xc = torch.from_numpy(x[local_idx][None])        # add channel dim
        yc = torch.tensor(y[local_idx])
        return xc, yc

# --------------------------------------------------------------------------- #
#                     Build global sample-level index                         #
# --------------------------------------------------------------------------- #
def build_sample_index() -> tuple[np.ndarray, np.ndarray]:
    """Return train_index, val_index where each row = (file_idx, local_idx)."""
    mapping = []
    for fi, meta in enumerate(INDEX["files"]):
        mapping.extend([(fi, li) for li in range(meta["n"])])
    mapping = np.asarray(mapping, dtype=np.int32)

    # stratified shuffle
    lbl = np.array([INDEX["files"][fi]["label"] for fi,_ in mapping])
    perm = np.random.permutation(len(mapping))
    mapping = mapping[perm]
    lbl     = lbl[perm]

    val_size = int(len(mapping) * VAL_SPLIT)
    train_idx, val_idx = mapping[val_size:], mapping[:val_size]
    return train_idx, val_idx

TRAIN_IDX, VAL_IDX = build_sample_index()

# --------------------------------------------------------------------------- #
#                           DataLoaders                                       #
# --------------------------------------------------------------------------- #
def make_loader(indices, shuffle) -> td.DataLoader:
    dataset = PeakDataset(indices)
    # Use the total batch size for multi-GPU training
    return td.DataLoader(
        dataset,
        batch_size=TOTAL_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0
    )

train_loader = make_loader(TRAIN_IDX, shuffle=True)
val_loader   = make_loader(VAL_IDX,   shuffle=False)

# --------------------------------------------------------------------------- #
#                                Model                                        #
# --------------------------------------------------------------------------- #
class PeakCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ch, fc, drop = PRESET["ch"], PRESET["fc"], PRESET["drop"]

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.ReLU(),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), nn.ReLU(),
            nn.Conv2d(ch[2], ch[3], 3, padding=1), nn.BatchNorm2d(ch[3]), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(ch[3], ch[3], 3, padding=1), nn.BatchNorm2d(ch[3]), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[3], fc[0]), nn.ReLU(), nn.Dropout(drop[0]),
            nn.Linear(fc[0], fc[1]), nn.ReLU(), nn.Dropout(drop[1]),
            nn.Linear(fc[1], 1)
        )

    def forward(self, x):
        return self.cls(self.conv(x)).squeeze(1)      # logits

# --------------------------------------------------------------------------- #
#                       Loss (BCE or focal)                                   #
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        prob     = torch.sigmoid(logits)
        p_t      = prob*target + (1-prob)*(1-target)
        focal    = (1-p_t)**self.gamma * bce_loss
        return focal.mean()

def make_loss():
    # class-imbalance weight = #neg / #pos
    num_pos = sum(f["n"] for f in INDEX["files"] if f["label"]==1)
    num_neg = INDEX["total"] - num_pos
    pos_w   = torch.tensor([num_neg/num_pos], dtype=torch.float32, device=device)
    if LOSS_FN == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)
    return FocalLoss(FOCAL_GAMMA, pos_weight=pos_w)

# --------------------------------------------------------------------------- #
#                         Initialize Model                                    #
# --------------------------------------------------------------------------- #
model = PeakCNN()

# Use DataParallel if multiple GPUs are available
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"Model wrapped with DataParallel using GPUs: {list(range(NUM_GPUS))}")

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
criterion = make_loss()

SAVE_DIR = Path(f"models_{MODEL_SIZE}_{time.strftime('%Y%m%d_%H%M%S')}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#                         Training utilities                                  #
# --------------------------------------------------------------------------- #
def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    t0 = time.time()
    total = correct = loss_sum = 0
    pbar = enumerate(loader)
    if train:
        pbar = tqdm(pbar, total=len(loader), ncols=90, desc="train")
    else:
        pbar = tqdm(pbar, total=len(loader), ncols=90, desc="val")
    
    for batch_idx, (x, y) in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        if train:
            optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        preds = (logits.sigmoid() > .5).float()
        correct += (preds == y).sum().item()
        loss_sum += loss.item() * len(x)
        total += len(x)

        if train and (batch_idx + 1) % PRINT_EVERY == 0:
            acc = correct / total
            speed = (time.time() - t0) / total
            eta = speed * (len(loader.dataset) - total)
            tqdm.write(f"  [{batch_idx+1:>5}/{len(loader)}] "
                       f"loss {loss_sum/total:.4f} acc {acc:.4f} "
                       f"ETA {eta/60:.1f}min")

    return loss_sum / total, correct / total

# --------------------------------------------------------------------------- #
#                                Training loop                                #
# --------------------------------------------------------------------------- #
print(f"\nStarting training with {NUM_GPUS} GPU(s)")
print(f"Total samples: {INDEX['total']:,}")
print(f"Train samples: {len(TRAIN_IDX):,}")
print(f"Val samples: {len(VAL_IDX):,}")
print(f"Batch size: {TOTAL_BATCH_SIZE} total ({BATCH_SIZE} per GPU)" if NUM_GPUS > 1 else f"Batch size: {BATCH_SIZE}")
print("-" * 60)

best_val = 0
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS} -- LR {optimizer.param_groups[0]['lr']:.3e}")

    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")
    
    scheduler.step(val_loss)

    # checkpoint
    ckpt = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "opt": optimizer.state_dict(),
        "val_acc": val_acc,
        "num_gpus": NUM_GPUS,
    }
    torch.save(ckpt, SAVE_DIR / f"epoch_{epoch:03d}.pth")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(ckpt, SAVE_DIR / "best.pth")
        print("  ✓ new best model saved")

print(f"\nTraining finished ▸ best val acc {best_val:.4f}")
print(f"Models saved to: {SAVE_DIR}")