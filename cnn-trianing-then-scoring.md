Currently the model training in CNN4-O3pro-opt_wsyDataset.py is not optimal;
First the training needs to take the data imbalance 3.4:1 decoy:target ratio into account;
then what we want the model to achieve is that:
it would score the target higher, and we want less decoy that is been scored very high;
we want a good FDR;

currently the model is scoring either target or decoy very low (in order to minimize the loss? i am not sure the reason why);

Also it seems like that the current model will experience overfitting after 15 epochs (very high training accuracy but low validation accuracy);

Help me re-design the model architecture, and re-design the training methods, to achieve a good FDR and score decoy Lower (We want decoy to be further from 1 as possible), but DO score target Higher (better target should be closer to 1);

Aslo help me adjust the training epoches down to 20 epoches;

After the training, we should use the best saved model to directly run the process in score_all_npy3.py, to score the npy files and save all the results, Make sure that you name the output directory correctly so that it could easily distinguish each run's result;

Give me a Complete Enhanced Code;

---

File: CNN4-O3pro-opt_wsyDataset.py
```py
#!/usr/bin/env python3
# ------------------------------------------------------------
# CNN-based peak-group classifier with AiStation-compatible data loading
# 
# Just run: python CNN4-O3pro-aistation.py
# It will automatically detect and use all available GPUs
# ------------------------------------------------------------
"""
Requirements
------------
pip install torch>=2.0 tqdm psutil
"""
from __future__ import annotations
import os, pickle, random, time, math, json, logging, sys
from pathlib import Path
from collections import OrderedDict, defaultdict
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset
from tqdm import tqdm, trange
import psutil

# --------------------------------------------------------------------------- #
#                            Setup Logging                                    #
# --------------------------------------------------------------------------- #
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.StreamHandler(sys.stderr)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --------------------------------------------------------------------------- #
#                            Configuration section                            #
# --------------------------------------------------------------------------- #

# Data paths - use semicolon to separate multiple paths
TRAIN_DATA_PATHS = ";".join([
    "/guotiannan/train_data_20250830/decoy_base_reverse0",
    "/guotiannan/train_data_20250830/decoy_base_rt_shift",
    "/guotiannan/train_data_20250830/diann20_target",
    "/guotiannan/train_data_20250830/diann20_target_rt_shift"
])

VAL_DATA_PATHS = TRAIN_DATA_PATHS  # Using same paths, but will split internally

LABELS = {  # folder → class label
    "decoy_base_reverse0": 0,
    "decoy_base_rt_shift": 0,
    "diann20_target": 1,
    "diann20_target_rt_shift": 1,
}

# Training configuration
config = {
    "epochs": 50,
    "train_batch_size": 1024,
    "predict_batch_size": 2048,
    "buffer_size": 16,  # Buffer size for shuffle
    "num_workers": 8,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "loss_fn": "bce",  # "bce" | "focal"
    "focal_gamma": 2.0,
    "model_size": "medium",  # see presets below
    "val_split": 0.02,  # 2% validation
    "seed": 42,
    "print_every": 50,  # batches
    "gpu_num": torch.cuda.device_count() if torch.cuda.is_available() else 1,
}

logger.info("=" * 80)
logger.info("Starting CNN Training on AiStation")
logger.info("=" * 80)
logger.info(f"Configuration: {config}")

# --------------------------------------------------------------------------- #
#                          Dataset class definition                           #
# --------------------------------------------------------------------------- #
class Dataset(td.Dataset):
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
            "frag_info": self.frag_info[idx] if self.frag_info is not None else np.zeros(1),
            "feat": self.feat[idx] if self.feat is not None else np.zeros(1),
            "label": self.label[idx],
            "file": self.file[idx] if self.file is not None else "",
            "precursor_id": self.precursor_id[idx] if self.precursor_id is not None else 0,
        }

    def __len__(self):
        return len(self.rsm) if self.rsm is not None else 0

# --------------------------------------------------------------------------- #
#                       Model-complexity presets                              #
# --------------------------------------------------------------------------- #
PRESET = {
    "small": dict(ch=[16, 32, 64, 128], fc=[128, 64], drop=[0.3, 0.2]),
    "medium": dict(ch=[32, 64, 128, 256], fc=[256, 128], drop=[0.5, 0.3]),
    "large": dict(ch=[64, 128, 256, 512], fc=[512, 256], drop=[0.5, 0.3]),
    "xlarge": dict(ch=[128, 256, 512, 1024], fc=[1024, 512], drop=[0.5, 0.4]),
}[config["model_size"]]

# --------------------------------------------------------------------------- #
#                          Reproducibility                                    #
# --------------------------------------------------------------------------- #
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(config["seed"])

# --------------------------------------------------------------------------- #
#                          Auto GPU Detection                                 #
# --------------------------------------------------------------------------- #
def setup_device():
    """Automatically detect and setup GPUs"""
    if not torch.cuda.is_available():
        logger.info("No CUDA devices found. Using CPU.")
        return torch.device("cpu"), 0
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s) available:")
    for i in range(num_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Use first GPU as primary device
    device = torch.device("cuda:0")
    return device, num_gpus

device, NUM_GPUS = setup_device()

# --------------------------------------------------------------------------- #
#                    AiStation-compatible Data Loading                        #
# --------------------------------------------------------------------------- #

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def collate_batch_cnn(batch_data):
    """Collate batch of samples for CNN training."""
    rsm_list = []
    label_list = []
    
    for batch in batch_data:
        rsm = batch["rsm"]
        # Process rsm: (72, 8, 16) → mean first 5 frag dims → (72, 16)
        if rsm.shape[1] >= 5:
            rsm_processed = rsm[:, :5, :].mean(axis=1)
        else:
            rsm_processed = rsm.mean(axis=1)
        
        rsm_list.append(rsm_processed)
        label_list.append(batch["label"])
    
    # Convert to tensors
    one_batch_rsm = torch.tensor(np.array(rsm_list), dtype=torch.float32)
    one_batch_label = torch.tensor(np.array(label_list), dtype=torch.float32)
    
    # Add channel dimension for CNN: (batch, 72, 16) -> (batch, 1, 72, 16)
    one_batch_rsm = one_batch_rsm.unsqueeze(1)
    
    # Handle NaN values
    one_batch_rsm = torch.nan_to_num(one_batch_rsm)
    one_batch_label = torch.nan_to_num(one_batch_label)
    
    return one_batch_rsm, one_batch_label

def shuffle_file_list(file_list, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(file_list), generator=generator).numpy()
    file_list = (np.array(file_list)[idx]).tolist()
    return file_list

class IterableDiartDataset(IterableDataset):
    """Custom dataset class for efficient data loading on AiStation."""
    
    def __init__(self,
                 file_list: list,
                 file_bin_dict=None,
                 batch_size=1024,
                 buffer_size=2,
                 gpu_num=1,
                 shuffle=True,
                 seed=0,
                 multi_node=False):
        super(IterableDiartDataset).__init__()
        self.file_list = file_list
        self.file_bin_dict = file_bin_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.gpu_num = gpu_num
        self.multi_node = multi_node
        self.buffer_size = buffer_size
        
    def parse_file(self, file_name):
        if self.file_bin_dict is not None:
            data = []
            for bin_file in file_name:
                try:
                    with open(bin_file, "rb") as f:
                        ds = pickle.load(f)
                        if not isinstance(ds, Dataset):
                            logger.warning(f"File {bin_file} is not a Dataset object")
                            continue
                        data.append(ds)
                except Exception as e:
                    logger.error(f'Load {bin_file} error: {e}')
                    continue
            if not data:
                return None
            data = ConcatDataset(data)
        else:
            try:
                with open(file_name, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                logger.error(f'Load {file_name} error: {e}')
                return None
        
        return DataLoader(data,
                          shuffle=False,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=0,
                          collate_fn=collate_batch_cnn)
    
    def file_mapper(self, file_list):
        idx = 0
        file_num = len(file_list)
        while idx < file_num:
            if self.file_bin_dict is not None:
                loader = self.parse_file(self.file_bin_dict[file_list[idx]])
            else:
                loader = self.parse_file(file_list[idx])
            
            if loader is not None:
                yield loader
            idx += 1
    
    def __iter__(self):
        if self.gpu_num > 1:
            if self.multi_node:  # Multi-node multi-GPU
                rank = int(os.environ.get('RANK', 0))
                file_itr = self.file_list[rank::self.gpu_num]
            else:  # Single-node multi-GPU
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                file_itr = self.file_list[local_rank::self.gpu_num]
        else:
            file_itr = self.file_list
        
        file_mapped_itr = self.file_mapper(file_itr)
        
        if self.shuffle:
            return self._shuffle(file_mapped_itr)
        else:
            return file_mapped_itr
    
    def __len__(self):
        if self.gpu_num > 1:
            return math.ceil(len(self.file_list) / self.gpu_num)
        else:
            return len(self.file_list)
    
    def _shuffle(self, mapped_itr):
        buffer = []
        for dt in mapped_itr:
            if len(buffer) < self.buffer_size:
                buffer.append(dt)
            else:
                i = random.randint(0, self.buffer_size - 1)
                yield buffer[i]
                buffer[i] = dt
        random.shuffle(buffer)
        yield from buffer

def create_iterable_dataset(data_path, config, parse='train', multi_node=False):
    """Create iterable dataset for efficient data loading on AiStation."""
    
    logger.info(f"Creating {parse} dataset from: {data_path}")
    
    if parse == 'train':
        total_train_path = data_path.split(';')
        train_file_list = []
        
        for train_path in total_train_path:
            train_part_file_list = glob.glob(f'{train_path}/*.pkl')
            logger.info(f"Found {len(train_part_file_list)} files in {train_path}")
            
            train_part_file_list_clean = [f for f in train_part_file_list if not is_file_empty(f)]
            logger.info(f"After filtering empty files: {len(train_part_file_list_clean)} files")
            
            if len(train_part_file_list_clean) > 0:
                train_file_list.extend(train_part_file_list_clean)
        
        random.shuffle(train_file_list)
        train_file_list = shuffle_file_list(train_file_list, config['seed'])
        
        logger.info(f"Total train files loaded: {len(train_file_list)}")
        
        # Split for validation
        val_size = int(len(train_file_list) * config['val_split'])
        val_file_list = train_file_list[:val_size]
        train_file_list = train_file_list[val_size:]
        
        logger.info(f"After train/val split: Train={len(train_file_list)}, Val={len(val_file_list)}")
        
        # Update GPU number
        gpu_num = config['gpu_num'] if multi_node else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        
        # Truncate for multi-GPU training
        file_bin_num = len(train_file_list) // (3 * gpu_num)
        file_truncation_num = file_bin_num * (3 * gpu_num)
        train_file_list = train_file_list[:file_truncation_num]
        
        logger.info(f"After truncation for {gpu_num} GPUs: {len(train_file_list)} files")
        
        # Create file bins
        file_bin_dict = defaultdict(list)
        for i in range(len(train_file_list)):
            file_bin_dict[i // 3].append(train_file_list[i])
        file_bin_list = list(file_bin_dict.keys())
        
        train_dl = IterableDiartDataset(
            file_bin_list,
            file_bin_dict=file_bin_dict,
            batch_size=config["train_batch_size"],
            buffer_size=config["buffer_size"],
            gpu_num=gpu_num,
            shuffle=True,
            seed=config['seed'],
            multi_node=multi_node
        )
        
        # Create validation dataset
        val_bin_dict = defaultdict(list)
        for i in range(len(val_file_list)):
            val_bin_dict[i].append(val_file_list[i])
        val_bin_list = list(val_bin_dict.keys())
        
        val_dl = IterableDiartDataset(
            val_bin_list,
            file_bin_dict=val_bin_dict,
            batch_size=config["predict_batch_size"],
            buffer_size=1,
            gpu_num=gpu_num,
            shuffle=False,
            seed=config['seed'],
            multi_node=multi_node
        )
        
        logger.info(f"Dataset created: Train bins={len(file_bin_list)}, Val bins={len(val_bin_list)}")
        logger.info(f"Batch sizes: Train={config['train_batch_size']}, Val={config['predict_batch_size']}")
        
        return train_dl, val_dl
    
    else:  # validation only
        val_file_list = []
        if ';' in data_path:
            total_val_path = data_path.split(';')
            for val_path in total_val_path:
                val_part_file_list = glob.glob(f'{val_path}/*.pkl')
                val_part_file_list_clean = [f for f in val_part_file_list if not is_file_empty(f)]
                if len(val_part_file_list_clean) > 0:
                    val_file_list.extend(val_part_file_list_clean)
        else:
            val_file_list = glob.glob(f'{data_path}/*.pkl')
            val_file_list = [f for f in val_file_list if not is_file_empty(f)]
        
        gpu_num = config['gpu_num'] if multi_node else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        
        file_bin_dict = defaultdict(list)
        for i in range(len(val_file_list)):
            file_bin_dict[i].append(val_file_list[i])
        file_bin_list = list(file_bin_dict.keys())
        
        val_dl = IterableDiartDataset(
            file_bin_list,
            file_bin_dict=file_bin_dict,
            batch_size=config["predict_batch_size"],
            buffer_size=1,
            gpu_num=gpu_num,
            shuffle=False,
            multi_node=multi_node
        )
        
        logger.info(f"Validation dataset created: {len(val_file_list)} files")
        
        return val_dl

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
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), nn.ReLU(),
            nn.Conv2d(ch[2], ch[3], 3, padding=1), nn.BatchNorm2d(ch[3]), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
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
        return self.cls(self.conv(x)).squeeze(1)  # logits

# --------------------------------------------------------------------------- #
#                       Loss (BCE or focal)                                   #
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = (1 - p_t) ** self.gamma * bce_loss
        return focal.mean()

def make_loss():
    # For now, using balanced weight
    pos_w = torch.tensor([1.0], dtype=torch.float32, device=device)
    if config["loss_fn"] == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)
    return FocalLoss(config["focal_gamma"], pos_weight=pos_w)

# --------------------------------------------------------------------------- #
#                       Create DataLoaders                                    #
# --------------------------------------------------------------------------- #
logger.info("Creating datasets...")
train_dataset, val_dataset = create_iterable_dataset(
    TRAIN_DATA_PATHS, 
    config, 
    parse='train',
    multi_node=False
)

# --------------------------------------------------------------------------- #
#                         Initialize Model                                    #
# --------------------------------------------------------------------------- #
logger.info("Initializing model...")
model = PeakCNN()

# Use DataParallel if multiple GPUs are available
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    logger.info(f"Model wrapped with DataParallel using GPUs: {list(range(NUM_GPUS))}")

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
criterion = make_loss()

SAVE_DIR = Path(f"models_{config['model_size']}_{time.strftime('%Y%m%d_%H%M%S')}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Models will be saved to: {SAVE_DIR}")

# --------------------------------------------------------------------------- #
#                         Training utilities                                  #
# --------------------------------------------------------------------------- #
def run_epoch(dataset, train: bool, epoch_num: int):
    if train:
        model.train()
        phase = "train"
    else:
        model.eval()
        phase = "val"
    
    t0 = time.time()
    total = correct = loss_sum = 0
    batch_count = 0
    
    logger.info(f"Starting {phase} epoch {epoch_num}...")
    
    # Iterate through file loaders
    for file_loader_idx, file_loader in enumerate(dataset):
        file_batch_count = 0
        file_total = 0
        
        for batch_idx, (x, y) in enumerate(file_loader):
            if x.numel() == 0:  # Skip empty batches
                continue
                
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
            file_total += len(x)
            batch_count += 1
            file_batch_count += 1
            
            if batch_count % config["print_every"] == 0:
                acc = correct / total if total > 0 else 0
                avg_loss = loss_sum / total if total > 0 else 0
                speed = (time.time() - t0) / total if total > 0 else 0
                logger.info(
                    f"  {phase} Epoch {epoch_num} | File {file_loader_idx+1} | "
                    f"Batch {batch_count} | Samples {total} | "
                    f"Loss {avg_loss:.4f} | Acc {acc:.4f} | "
                    f"Speed {speed:.3f}s/sample"
                )
        
        if file_batch_count > 0:
            logger.info(f"  Completed file {file_loader_idx+1}: {file_total} samples processed")
    
    if total == 0:
        logger.warning(f"No samples processed in {phase} epoch!")
        return 0.0, 0.0
    
    return loss_sum / total, correct / total

# --------------------------------------------------------------------------- #
#                                Training loop                                #
# --------------------------------------------------------------------------- #
logger.info("=" * 80)
logger.info("Starting training")
logger.info(f"Configuration:")
logger.info(f"  - GPUs: {NUM_GPUS}")
logger.info(f"  - Batch size: {config['train_batch_size']}")
logger.info(f"  - Learning rate: {config['lr']}")
logger.info(f"  - Epochs: {config['epochs']}")
logger.info(f"  - Model size: {config['model_size']}")
logger.info("=" * 80)

best_val = 0
for epoch in range(1, config["epochs"] + 1):
    logger.info(f"\n{'='*60}")
    logger.info(f"Epoch {epoch}/{config['epochs']} -- LR {optimizer.param_groups[0]['lr']:.3e}")
    logger.info(f"{'='*60}")
    
    tr_loss, tr_acc = run_epoch(train_dataset, train=True, epoch_num=epoch)
    logger.info(f"Train epoch {epoch} completed: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
    
    val_loss, val_acc = run_epoch(val_dataset, train=False, epoch_num=epoch)
    logger.info(f"Val epoch {epoch} completed: loss={val_loss:.4f}, acc={val_acc:.4f}")
    
    logger.info(f"Epoch {epoch} Summary: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    scheduler.step(val_loss)
    
    # Save checkpoint
    ckpt = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "opt": optimizer.state_dict(),
        "val_acc": val_acc,
        "train_acc": tr_acc,
        "val_loss": val_loss,
        "train_loss": tr_loss,
        "num_gpus": NUM_GPUS,
    }
    
    ckpt_path = SAVE_DIR / f"epoch_{epoch:03d}.pth"
    torch.save(ckpt, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")
    
    if val_acc > best_val:
        best_val = val_acc
        best_path = SAVE_DIR / "best.pth"
        torch.save(ckpt, best_path)
        logger.info(f"✓ New best model saved to {best_path} (val_acc={val_acc:.4f})")

logger.info("=" * 80)
logger.info(f"Training finished! Best validation accuracy: {best_val:.4f}")
logger.info(f"All models saved to: {SAVE_DIR}")
logger.info("=" * 80)
```

File: score_all_npy3.py
```py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime
import os

# Configuration
MODEL_PATH = '/wangshuaiyao/jiangheng/dia-cnn/dia-cnn-aistation/model_epoch_004.pth'
print(f'load successful: {MODEL_PATH}')

DATA_FOLDER = '/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm/output_new'
print(f'load successful: {DATA_FOLDER}')

# Create output directory with timestamp
OUTPUT_DIR = f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Output directory created: {OUTPUT_DIR}')

MODEL_COMPLEXITY = "medium"
SAMPLES_PER_BATCH = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Model complexity: {MODEL_COMPLEXITY}, Samples per batch: {SAMPLES_PER_BATCH}, Device: {device}')

# Complexity presets
COMPLEXITY_CONFIG = {
    "small": {
        "channels": [16, 32, 64, 128],
        "fc_dims": [128, 64],
        "dropout": [0.3, 0.2]
    },
    "medium": {
        "channels": [32, 64, 128, 256],
        "fc_dims": [256, 128],
        "dropout": [0.5, 0.3]
    },
    "large": {
        "channels": [64, 128, 256, 512],
        "fc_dims": [512, 256],
        "dropout": [0.5, 0.3]
    },
    "xlarge": {
        "channels": [128, 256, 512, 1024],
        "fc_dims": [1024, 512],
        "dropout": [0.5, 0.4]
    }
}

# Model Definition
class PeakGroupCNN(nn.Module):
    def __init__(self, complexity="medium"):
        super().__init__()
        
        config = COMPLEXITY_CONFIG[complexity]
        channels = config["channels"]
        fc_dims = config["fc_dims"]
        dropout_rates = config["dropout"]
        
        # First conv block
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.pool1 = nn.MaxPool2d((2, 2))
        
        # Second conv block
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(channels[3])
        self.pool2 = nn.MaxPool2d((2, 2))
        
        # Additional conv layer
        self.conv5 = nn.Conv2d(channels[3], channels[3], kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(channels[3])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(channels[3], fc_dims[0])
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.fc3 = nn.Linear(fc_dims[1], 1)
        
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Conv block 3
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global pooling and FC layers
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze(1)

def prepare_windows(rsm_data, rt_values, samples_per_batch, window_size=16):
    """Prepare windows from RSM data"""
    # Sum across channels and smooth
    aggregated = np.sum(rsm_data, axis=1)
    smoothed = gaussian_filter1d(aggregated, sigma=1, axis=-1)
    
    # Aggregate RT points (factor of 3)
    n_samples, n_mz, n_rt = smoothed.shape
    agg_factor = 3
    n_agg_rt = n_rt // agg_factor
    
    all_windows = []
    window_info = []
    
    for sample_idx in range(min(samples_per_batch, n_samples)):
        sample_rsm = smoothed[sample_idx]
        
        # Aggregate
        agg_rsm = np.zeros((n_mz, n_agg_rt))
        for i in range(n_agg_rt):
            start = i * agg_factor
            agg_rsm[:, i] = np.mean(sample_rsm[:, start:start+agg_factor], axis=1)
        
        # Create windows
        max_windows = min(100, n_agg_rt - window_size + 1)
        for w_idx in range(max_windows):
            window = agg_rsm[:, w_idx:w_idx + window_size]
            # Normalize
            w_min, w_max = window.min(), window.max()
            if w_max > w_min:
                window = (window - w_min) / (w_max - w_min)
            else:
                window = np.zeros_like(window)
            
            all_windows.append(window)
            window_info.append((sample_idx, w_idx))
    
    return np.array(all_windows) if all_windows else np.array([]), window_info

def process_single_batch(batch_num, folder_path, model, device):
    """Process a single batch and return results"""
    
    # Define file paths
    index_file = folder_path / f"batch_{batch_num}_index.txt"
    rsm_file = folder_path / f"batch_{batch_num}_rsm.npy"
    rt_file = folder_path / f"batch_{batch_num}_rt_values.npy"
    
    # Check if files exist
    if not (index_file.exists() and rsm_file.exists() and rt_file.exists()):
        return None
    
    # Load data
    rsm_data = np.load(rsm_file)
    rt_values = np.load(rt_file)
    
    # Load metadata
    with open(index_file, 'r') as f:
        lines = f.readlines()
    
    precursor_ids = []
    is_decoy = []
    for line in lines:
        parts = line.strip().split('\t') if '\t' in line else line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            precursor_id = parts[1]
            precursor_ids.append(precursor_id)
            is_decoy.append('DECOY' in precursor_id.upper() or precursor_id.startswith('rev_'))
    
    # Process windows
    windows, window_info = prepare_windows(rsm_data, rt_values, SAMPLES_PER_BATCH)
    
    if len(windows) == 0:
        return []
    
    # Add channel dimension
    windows = np.expand_dims(windows, axis=1)
    
    # Score windows
    batch_size = 512
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            scores = model(batch).cpu().numpy()
            all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    
    # Find best window for each sample
    sample_results = {}
    for (sample_idx, w_idx), score in zip(window_info, all_scores):
        if sample_idx not in sample_results or score > sample_results[sample_idx]['score']:
            sample_results[sample_idx] = {
                'score': float(score),
                'window_idx': int(w_idx),
                'is_decoy': is_decoy[sample_idx] if sample_idx < len(is_decoy) else False,
                'precursor_id': precursor_ids[sample_idx] if sample_idx < len(precursor_ids) else f"Sample_{sample_idx}",
                'batch_num': batch_num
            }
    
    return list(sample_results.values())

# MAIN EXECUTION
print("="*70)
print("CNN MODEL - PROCESSING ALL BATCHES")
print("="*70)

# Load model
print("\nLoading model...")
model = PeakGroupCNN(complexity=MODEL_COMPLEXITY).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
else:
    model.load_state_dict(checkpoint)
    print("Model loaded successfully")
model.eval()

# Find maximum batch number
folder_path = Path(DATA_FOLDER)
max_batch = -1
for file in folder_path.glob("batch_*_index.txt"):
    try:
        batch_num = int(file.stem.split('_')[1])
        if batch_num > max_batch:
            max_batch = batch_num
    except:
        continue

if max_batch == -1:
    print("No batch files found!")
    exit(1)

print(f"\nFound batches from 0 to {max_batch}")
print("Starting batch processing...")
print("-" * 50)

# Process all batches
successful_batches = 0
failed_batches = []
total_samples = 0

# Process batches from 0 to max_batch
for batch_num in range(max_batch + 1):
    # Show progress every 10 batches
    if batch_num % 10 == 0:
        print(f"Processing batch {batch_num}...")
    
    # Process the batch
    results = process_single_batch(batch_num, folder_path, model, device)
    
    if results is not None:
        successful_batches += 1
        total_samples += len(results)
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        # Calculate statistics
        n_targets = sum(1 for r in results if not r['is_decoy'])
        n_decoys = len(results) - n_targets
        
        # Prepare output data
        output_data = {
            'batch_num': batch_num,
            'n_samples': len(results),
            'n_targets': n_targets,
            'n_decoys': n_decoys,
            'target_rate': n_targets / len(results) if len(results) > 0 else 0,
            'max_score': max(r['score'] for r in results) if results else 0,
            'min_score': min(r['score'] for r in results) if results else 0,
            'mean_score': sum(r['score'] for r in results) / len(results) if results else 0,
            'results': results  # Full results for this batch
        }
        
        # Save to individual file
        output_file = os.path.join(OUTPUT_DIR, f'batch_{batch_num:04d}_results.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Show detailed progress every 50 batches
        if batch_num % 50 == 0 and batch_num > 0:
            print(f"  Progress: {batch_num}/{max_batch} | Successful: {successful_batches} | Total samples: {total_samples}")
    else:
        failed_batches.append(batch_num)

print("-" * 50)
print("\nProcessing complete!")
print(f"Successful batches: {successful_batches}")
print(f"Failed/missing batches: {len(failed_batches)}")
if failed_batches and len(failed_batches) <= 20:
    print(f"Failed batch numbers: {failed_batches}")
print(f"Total samples processed: {total_samples}")

# Create summary file
summary = {
    'processing_date': datetime.now().isoformat(),
    'model_path': MODEL_PATH,
    'data_folder': DATA_FOLDER,
    'output_directory': OUTPUT_DIR,
    'total_batches': max_batch + 1,
    'successful_batches': successful_batches,
    'failed_batches': failed_batches,
    'total_samples': total_samples,
    'device': device
}

summary_file = os.path.join(OUTPUT_DIR, '_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll results saved to directory: {OUTPUT_DIR}")
print(f"Summary saved to: {summary_file}")
print("="*70)
```

