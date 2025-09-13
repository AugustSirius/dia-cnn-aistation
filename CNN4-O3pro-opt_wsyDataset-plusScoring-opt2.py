#!/usr/bin/env python3
# ------------------------------------------------------------
# Enhanced CNN-based peak-group classifier with FDR optimization
# Includes automatic scoring after training completes
# ------------------------------------------------------------
"""
Requirements
------------
pip install torch>=2.0 tqdm psutil scikit-learn scipy
"""
from __future__ import annotations
import os, pickle, random, time, math, json, logging, sys
from pathlib import Path
from collections import OrderedDict, defaultdict
import glob
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset
from tqdm import tqdm, trange
import psutil
from sklearn.metrics import roc_auc_score

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

# Data paths for training
TRAIN_DATA_PATHS = ";".join([
    "/guotiannan/train_data_20250830/decoy_base_reverse0",
    "/guotiannan/train_data_20250830/decoy_base_rt_shift",
    "/guotiannan/train_data_20250830/diann20_target",
    "/guotiannan/train_data_20250830/diann20_target_rt_shift"
])

VAL_DATA_PATHS = TRAIN_DATA_PATHS  # Using same paths, but will split internally

# Data path for scoring
SCORING_DATA_FOLDER = '/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm/output_new'

LABELS = {  # folder → class label
    "decoy_base_reverse0": 0,
    "decoy_base_rt_shift": 0,
    "diann20_target": 1,
    "diann20_target_rt_shift": 1,
}

# Class imbalance ratio (decoy:target = 3.4:1)
CLASS_IMBALANCE_RATIO = 3.4

# Training configuration
config = {
    "epochs": 20,  # Reduced from 50 to prevent overfitting
    "train_batch_size": 1024,
    "predict_batch_size": 2048,
    "buffer_size": 16,
    "num_workers": 8,
    "lr": 1e-4,  # Reduced learning rate
    "weight_decay": 3e-4,  # Increased for better regularization
    "loss_fn": "margin_bce",  # Using margin BCE for better separation
    "margin": 2.0,  # Margin for margin BCE loss
    "ranking_weight": 0.3,  # Weight for ranking loss component
    "focal_gamma": 2.0,
    "model_size": "medium",
    "val_split": 0.02,  # 2% validation
    "seed": 42,
    "print_every": 50,
    "early_stop_patience": 4,  # Early stopping patience
    "gpu_num": torch.cuda.device_count() if torch.cuda.is_available() else 1,
    "auto_score_after_training": True,  # Automatically run scoring after training
    "samples_per_batch_scoring": 1000,  # For scoring phase
}

logger.info("=" * 80)
logger.info("Enhanced CNN Training with FDR Optimization + Auto Scoring")
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
    "small": dict(ch=[16, 32, 64, 128], fc=[128, 64], drop=[0.3, 0.2, 0.2]),
    "medium": dict(ch=[32, 64, 128, 256], fc=[256, 128], drop=[0.5, 0.3, 0.3]),
    "large": dict(ch=[64, 128, 256, 512], fc=[512, 256], drop=[0.5, 0.3, 0.3]),
    "xlarge": dict(ch=[128, 256, 512, 1024], fc=[1024, 512], drop=[0.5, 0.4, 0.4]),
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
            if self.multi_node:
                rank = int(os.environ.get('RANK', 0))
                file_itr = self.file_list[rank::self.gpu_num]
            else:
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
#                          Enhanced Model Architecture                        #
# --------------------------------------------------------------------------- #
class EnhancedPeakCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ch, fc, drop = PRESET["ch"], PRESET["fc"], PRESET["drop"]
        
        # Initialize bias for class imbalance
        self.init_bias = math.log(1.0 / CLASS_IMBALANCE_RATIO)
        
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, ch[0], 3, padding=1), 
            nn.BatchNorm2d(ch[0]), 
            nn.SiLU(),  # Using SiLU instead of ReLU
            nn.Conv2d(ch[0], ch[1], 3, padding=1), 
            nn.BatchNorm2d(ch[1]), 
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(ch[1], ch[2], 3, padding=1), 
            nn.BatchNorm2d(ch[2]), 
            nn.SiLU(),
            nn.Conv2d(ch[2], ch[3], 3, padding=1), 
            nn.BatchNorm2d(ch[3]), 
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(drop[2]),  # Added spatial dropout after second pool
            
            # Block 3
            nn.Conv2d(ch[3], ch[3], 3, padding=1), 
            nn.BatchNorm2d(ch[3]), 
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop[0]),  # Dropout before first FC
            nn.Linear(ch[3], fc[0]), 
            nn.SiLU(), 
            nn.Dropout(drop[0]),
            nn.Linear(fc[0], fc[1]), 
            nn.SiLU(), 
            nn.Dropout(drop[1]),
            nn.Linear(fc[1], 1)
        )
        
        # Initialize final layer bias for class imbalance
        self.cls[-1].bias.data.fill_(self.init_bias)
    
    def forward(self, x):
        return self.cls(self.conv(x)).squeeze(1)  # logits

# --------------------------------------------------------------------------- #
#                          Enhanced Loss Functions                            #
# --------------------------------------------------------------------------- #
class MarginBCELoss(nn.Module):
    """BCE loss with margin to push scores further apart"""
    def __init__(self, margin=2.0, pos_weight=None):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        # Add margin: push targets higher, decoys lower
        logits_shifted = torch.where(
            targets == 1,
            logits - self.margin,  # For targets, subtract margin (need higher logits)
            logits + self.margin   # For decoys, add margin (need lower logits)
        )
        
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits_shifted, targets, pos_weight=self.pos_weight, reduction='mean'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits_shifted, targets, reduction='mean'
            )
        return loss

class RankingLoss(nn.Module):
    """Pairwise ranking loss to ensure targets score higher than decoys"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, logits, targets):
        # Separate target and decoy scores
        target_mask = targets == 1
        decoy_mask = targets == 0
        
        target_scores = logits[target_mask]
        decoy_scores = logits[decoy_mask]
        
        if len(target_scores) == 0 or len(decoy_scores) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Sample pairs
        n_pairs = min(len(target_scores), len(decoy_scores))
        target_scores_sampled = target_scores[:n_pairs]
        decoy_scores_sampled = decoy_scores[:n_pairs]
        
        # Compute ranking loss
        loss = F.relu(self.margin - (target_scores_sampled - decoy_scores_sampled))
        return loss.mean()

class CombinedLoss(nn.Module):
    """Combine margin BCE with ranking loss"""
    def __init__(self, margin_bce=2.0, margin_rank=1.0, rank_weight=0.3, pos_weight=None):
        super().__init__()
        self.margin_bce = MarginBCELoss(margin_bce, pos_weight)
        self.ranking = RankingLoss(margin_rank)
        self.rank_weight = rank_weight
    
    def forward(self, logits, targets):
        bce_loss = self.margin_bce(logits, targets)
        rank_loss = self.ranking(logits, targets)
        return bce_loss + self.rank_weight * rank_loss

def make_loss():
    # Use class imbalance ratio as positive weight
    pos_weight = torch.tensor([CLASS_IMBALANCE_RATIO], dtype=torch.float32, device=device)
    
    if config["loss_fn"] == "margin_bce":
        return CombinedLoss(
            margin_bce=config["margin"],
            margin_rank=1.0,
            rank_weight=config["ranking_weight"],
            pos_weight=pos_weight
        )
    else:  # Standard BCE
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# --------------------------------------------------------------------------- #
#                          FDR Calculation and Metrics                        #
# --------------------------------------------------------------------------- #
def calculate_fdr_metrics(logits, targets, threshold=0.01):
    """Calculate FDR, recall at 1% FDR, and AUROC"""
    with torch.no_grad():
        scores = torch.sigmoid(logits).cpu().numpy()
        labels = targets.cpu().numpy()
        
        # AUROC
        try:
            auroc = roc_auc_score(labels, scores)
        except:
            auroc = 0.5
        
        # Sort by scores (high to low)
        order = scores.argsort()[::-1]
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        
        # Calculate FDR curve
        cum_decoys = (sorted_labels == 0).cumsum()
        cum_total = np.arange(1, len(scores) + 1)
        
        # Avoid division by zero
        fdr = np.where(cum_total > 0, cum_decoys / cum_total, 0)
        
        # Find recall at 1% FDR
        fdr_threshold = 0.01
        valid_indices = np.where(fdr <= fdr_threshold)[0]
        
        if len(valid_indices) > 0:
            cutoff_idx = valid_indices[-1]
            n_targets_at_cutoff = (sorted_labels[:cutoff_idx + 1] == 1).sum()
            total_targets = (labels == 1).sum()
            recall_at_1fdr = n_targets_at_cutoff / total_targets if total_targets > 0 else 0
            fdr_at_cutoff = fdr[cutoff_idx]
        else:
            recall_at_1fdr = 0.0
            fdr_at_cutoff = 1.0
        
        return {
            'auroc': auroc,
            'recall_at_1fdr': recall_at_1fdr,
            'fdr_at_cutoff': fdr_at_cutoff,
            'mean_target_score': scores[labels == 1].mean() if (labels == 1).sum() > 0 else 0,
            'mean_decoy_score': scores[labels == 0].mean() if (labels == 0).sum() > 0 else 0,
        }

# --------------------------------------------------------------------------- #
#                         Training utilities                                  #
# --------------------------------------------------------------------------- #
def run_epoch(dataset, model, optimizer, criterion, train: bool, epoch_num: int):
    if train:
        model.train()
        phase = "train"
    else:
        model.eval()
        phase = "val"
    
    t0 = time.time()
    total = correct = loss_sum = 0
    batch_count = 0
    
    all_logits = []
    all_targets = []
    
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # Store for metrics calculation
            if not train:
                all_logits.append(logits.detach())
                all_targets.append(y.detach())
            
            preds = (logits.sigmoid() > 0.5).float()
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
        return 0.0, 0.0, {}
    
    # Calculate FDR metrics for validation
    fdr_metrics = {}
    if not train and len(all_logits) > 0:
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        fdr_metrics = calculate_fdr_metrics(all_logits, all_targets)
        logger.info(f"  {phase} FDR Metrics: AUROC={fdr_metrics['auroc']:.4f}, "
                   f"Recall@1%FDR={fdr_metrics['recall_at_1fdr']:.4f}, "
                   f"Target_score={fdr_metrics['mean_target_score']:.4f}, "
                   f"Decoy_score={fdr_metrics['mean_decoy_score']:.4f}")
    
    return loss_sum / total, correct / total, fdr_metrics

# --------------------------------------------------------------------------- #
#                              Scoring Functions                              #
# --------------------------------------------------------------------------- #
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

def calculate_batch_fdr(results):
    """Calculate FDR metrics for a batch of results"""
    if not results:
        return {}
    
    # Sort by score (descending)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Calculate cumulative FDR
    cum_decoys = 0
    for i, result in enumerate(sorted_results):
        if result['is_decoy']:
            cum_decoys += 1
        cum_total = i + 1
        result['fdr'] = cum_decoys / cum_total if cum_total > 0 else 0
    
    # Find number of results at 1% FDR
    results_at_1fdr = 0
    for result in sorted_results:
        if result['fdr'] <= 0.01:
            results_at_1fdr += 1
        else:
            break
    
    n_targets = sum(1 for r in results if not r['is_decoy'])
    n_decoys = len(results) - n_targets
    
    return {
        'n_results_at_1fdr': results_at_1fdr,
        'n_targets': n_targets,
        'n_decoys': n_decoys,
        'target_rate': n_targets / len(results) if len(results) > 0 else 0
    }

def process_single_batch(batch_num, folder_path, model, device, samples_per_batch):
    """Process a single batch and return results with FDR analysis"""
    
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
    windows, window_info = prepare_windows(rsm_data, rt_values, samples_per_batch)
    
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
            logits = model(batch)
            scores = torch.sigmoid(logits).cpu().numpy()
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

def run_scoring(model_path, save_dir, model, config):
    """Run scoring on all batches using the trained model"""
    
    logger.info("\n" + "="*70)
    logger.info("STARTING AUTOMATIC SCORING PHASE")
    logger.info("="*70)
    
    # Create scoring output directory
    scoring_output_dir = save_dir / f"scoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    scoring_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scoring results will be saved to: {scoring_output_dir}")
    
    # Setup for scoring
    model.eval()
    folder_path = Path(SCORING_DATA_FOLDER)
    samples_per_batch = config['samples_per_batch_scoring']
    
    # Find maximum batch number
    max_batch = -1
    for file in folder_path.glob("batch_*_index.txt"):
        try:
            batch_num = int(file.stem.split('_')[1])
            if batch_num > max_batch:
                max_batch = batch_num
        except:
            continue
    
    if max_batch == -1:
        logger.warning("No batch files found for scoring!")
        return
    
    logger.info(f"Found batches from 0 to {max_batch}")
    logger.info("Starting batch processing...")
    logger.info("-" * 50)
    
    # Process all batches
    successful_batches = 0
    failed_batches = []
    total_samples = 0
    all_batch_results = []
    
    # Process batches from 0 to max_batch
    for batch_num in range(max_batch + 1):
        # Show progress every 10 batches
        if batch_num % 10 == 0:
            logger.info(f"Processing batch {batch_num}...")
        
        # Process the batch
        results = process_single_batch(batch_num, folder_path, model, device, samples_per_batch)
        
        if results is not None:
            successful_batches += 1
            total_samples += len(results)
            
            # Sort results by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Add rank
            for rank, result in enumerate(results, 1):
                result['rank'] = rank
            
            # Calculate FDR metrics
            fdr_metrics = calculate_batch_fdr(results)
            
            # Calculate statistics
            n_targets = fdr_metrics['n_targets']
            n_decoys = fdr_metrics['n_decoys']
            
            # Separate target and decoy scores
            target_scores = [r['score'] for r in results if not r['is_decoy']]
            decoy_scores = [r['score'] for r in results if r['is_decoy']]
            
            # Prepare output data
            output_data = {
                'batch_num': batch_num,
                'n_samples': len(results),
                'n_targets': n_targets,
                'n_decoys': n_decoys,
                'target_rate': fdr_metrics['target_rate'],
                'n_results_at_1fdr': fdr_metrics['n_results_at_1fdr'],
                'max_score': max(r['score'] for r in results) if results else 0,
                'min_score': min(r['score'] for r in results) if results else 0,
                'mean_score': sum(r['score'] for r in results) / len(results) if results else 0,
                'mean_target_score': sum(target_scores) / len(target_scores) if target_scores else 0,
                'mean_decoy_score': sum(decoy_scores) / len(decoy_scores) if decoy_scores else 0,
                'score_separation': (sum(target_scores) / len(target_scores) - 
                                   sum(decoy_scores) / len(decoy_scores)) if target_scores and decoy_scores else 0,
                'results': results  # Full results for this batch
            }
            
            all_batch_results.append(output_data)
            
            # Save to individual file
            output_file = scoring_output_dir / f'batch_{batch_num:04d}_results.json'
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Show detailed progress every 50 batches
            if batch_num % 50 == 0 and batch_num > 0:
                logger.info(f"  Progress: {batch_num}/{max_batch} | Successful: {successful_batches}")
                logger.info(f"  Total samples: {total_samples}")
                logger.info(f"  Last batch - Targets: {n_targets}, Decoys: {n_decoys}")
                logger.info(f"  Score separation: {output_data['score_separation']:.4f}")
        else:
            failed_batches.append(batch_num)
    
    logger.info("-" * 50)
    logger.info("\nScoring complete!")
    logger.info(f"Successful batches: {successful_batches}")
    logger.info(f"Failed/missing batches: {len(failed_batches)}")
    if failed_batches and len(failed_batches) <= 20:
        logger.info(f"Failed batch numbers: {failed_batches}")
    logger.info(f"Total samples processed: {total_samples}")
    
    # Calculate overall statistics
    if all_batch_results:
        overall_targets = sum(b['n_targets'] for b in all_batch_results)
        overall_decoys = sum(b['n_decoys'] for b in all_batch_results)
        overall_at_1fdr = sum(b['n_results_at_1fdr'] for b in all_batch_results)
        avg_score_separation = sum(b['score_separation'] for b in all_batch_results) / len(all_batch_results)
        
        logger.info("\nOverall Scoring Statistics:")
        logger.info(f"  Total targets: {overall_targets}")
        logger.info(f"  Total decoys: {overall_decoys}")
        logger.info(f"  Total results at 1% FDR: {overall_at_1fdr}")
        logger.info(f"  Average score separation: {avg_score_separation:.4f}")
    
    # Create summary file
    summary = {
        'processing_date': datetime.now().isoformat(),
        'model_path': str(model_path),
        'data_folder': SCORING_DATA_FOLDER,
        'output_directory': str(scoring_output_dir),
        'total_batches': max_batch + 1,
        'successful_batches': successful_batches,
        'failed_batches': failed_batches,
        'total_samples': total_samples,
        'device': str(device),
        'model_complexity': config['model_size'],
        'overall_statistics': {
            'total_targets': overall_targets if all_batch_results else 0,
            'total_decoys': overall_decoys if all_batch_results else 0,
            'total_at_1fdr': overall_at_1fdr if all_batch_results else 0,
            'avg_score_separation': avg_score_separation if all_batch_results else 0
        }
    }
    
    summary_file = scoring_output_dir / '_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nAll scoring results saved to directory: {scoring_output_dir}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("="*70)

# --------------------------------------------------------------------------- #
#                                MAIN EXECUTION                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ======================================================================= #
    #                           TRAINING PHASE                                #
    # ======================================================================= #
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: MODEL TRAINING")
    logger.info("="*80)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_iterable_dataset(
        TRAIN_DATA_PATHS, 
        config, 
        parse='train',
        multi_node=False
    )
    
    # Initialize model
    logger.info("Initializing enhanced model...")
    model = EnhancedPeakCNN()
    
    # Use DataParallel if multiple GPUs are available
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel using GPUs: {list(range(NUM_GPUS))}")
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='max')
    criterion = make_loss()
    
    SAVE_DIR = Path(f"models_{config['model_size']}_enhanced_{time.strftime('%Y%m%d_%H%M%S')}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {SAVE_DIR}")
    
    # Save configuration
    config_path = SAVE_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting enhanced training with FDR optimization")
    logger.info(f"Configuration:")
    logger.info(f"  - GPUs: {NUM_GPUS}")
    logger.info(f"  - Batch size: {config['train_batch_size']}")
    logger.info(f"  - Learning rate: {config['lr']}")
    logger.info(f"  - Weight decay: {config['weight_decay']}")
    logger.info(f"  - Epochs: {config['epochs']}")
    logger.info(f"  - Model size: {config['model_size']}")
    logger.info(f"  - Loss function: {config['loss_fn']}")
    logger.info(f"  - Class imbalance ratio: {CLASS_IMBALANCE_RATIO}")
    logger.info("=" * 80)
    
    best_recall_at_1fdr = 0
    best_auroc = 0
    early_stop_counter = 0
    best_model_path = None
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auroc': [],
        'val_recall_at_1fdr': [],
        'val_mean_target_score': [],
        'val_mean_decoy_score': []
    }
    
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config['epochs']} -- LR {optimizer.param_groups[0]['lr']:.3e}")
        logger.info(f"{'='*60}")
        
        tr_loss, tr_acc, _ = run_epoch(train_dataset, model, optimizer, criterion, train=True, epoch_num=epoch)
        logger.info(f"Train epoch {epoch} completed: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        
        val_loss, val_acc, val_metrics = run_epoch(val_dataset, model, optimizer, criterion, train=False, epoch_num=epoch)
        logger.info(f"Val epoch {epoch} completed: loss={val_loss:.4f}, acc={val_acc:.4f}")
        
        # Update training history
        training_history['train_loss'].append(tr_loss)
        training_history['train_acc'].append(tr_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        if val_metrics:
            training_history['val_auroc'].append(val_metrics['auroc'])
            training_history['val_recall_at_1fdr'].append(val_metrics['recall_at_1fdr'])
            training_history['val_mean_target_score'].append(val_metrics['mean_target_score'])
            training_history['val_mean_decoy_score'].append(val_metrics['mean_decoy_score'])
            
            # Update scheduler based on recall@1%FDR
            scheduler.step(val_metrics['recall_at_1fdr'])
        
        logger.info(f"Epoch {epoch} Summary: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "opt": optimizer.state_dict(),
            "val_acc": val_acc,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "train_loss": tr_loss,
            "val_metrics": val_metrics,
            "num_gpus": NUM_GPUS,
            "config": config,
        }
        
        ckpt_path = SAVE_DIR / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")
        
        # Save best model based on recall@1%FDR
        if val_metrics and val_metrics['recall_at_1fdr'] > best_recall_at_1fdr:
            best_recall_at_1fdr = val_metrics['recall_at_1fdr']
            best_model_path = SAVE_DIR / "best_recall_at_1fdr.pth"
            torch.save(ckpt, best_model_path)
            logger.info(f"✓ New best model (recall@1%FDR) saved to {best_model_path} "
                       f"(recall={best_recall_at_1fdr:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Save best model based on AUROC
        if val_metrics and val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_path = SAVE_DIR / "best_auroc.pth"
            torch.save(ckpt, best_path)
            logger.info(f"✓ New best model (AUROC) saved to {best_path} (auroc={best_auroc:.4f})")
        
        # Early stopping
        if early_stop_counter >= config['early_stop_patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Save training history
        history_path = SAVE_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"Training finished!")
    logger.info(f"Best recall@1%FDR: {best_recall_at_1fdr:.4f}")
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info(f"All models saved to: {SAVE_DIR}")
    logger.info("=" * 80)
    
    # ======================================================================= #
    #                           SCORING PHASE                                 #
    # ======================================================================= #
    
    if config['auto_score_after_training'] and best_model_path is not None:
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: AUTOMATIC SCORING WITH BEST MODEL")
        logger.info("="*80)
        
        # Load best model for scoring
        logger.info(f"Loading best model from: {best_model_path}")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        
        # Create a fresh model for scoring (without DataParallel)
        scoring_model = EnhancedPeakCNN().to(device)
        
        # Load the state dict
        if isinstance(model, nn.DataParallel):
            scoring_model.load_state_dict(best_checkpoint['model'])
        else:
            scoring_model.load_state_dict(best_checkpoint['model'])
        
        scoring_model.eval()
        
        # Run scoring
        run_scoring(best_model_path, SAVE_DIR, scoring_model, config)
        
        logger.info("\n" + "="*80)
        logger.info("ALL PROCESSING COMPLETE!")
        logger.info(f"Training results saved to: {SAVE_DIR}")
        logger.info(f"Best model: {best_model_path}")
        logger.info("="*80)
    else:
        if not config['auto_score_after_training']:
            logger.info("\nAutomatic scoring disabled. To run scoring manually, use the best model at:")
            logger.info(f"{best_model_path}")
        else:
            logger.warning("\nNo best model found for scoring!")