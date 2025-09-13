#!/usr/bin/env python3
# ------------------------------------------------------------
# Enhanced CNN-based peak-group classifier with FDR optimization
# Integrated training and scoring pipeline
# ------------------------------------------------------------

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

# Scoring data path
SCORING_DATA_FOLDER = '/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm/output_new'

# Enhanced training configuration for better FDR
config = {
    "epochs": 20,  # Reduced from 50 to prevent overfitting
    "train_batch_size": 512,  # Smaller batch for better gradient updates
    "predict_batch_size": 1024,
    "buffer_size": 16,
    "num_workers": 8,
    "lr": 5e-4,  # Higher initial learning rate
    "weight_decay": 5e-5,  # Increased regularization
    "loss_fn": "focal_balanced",  # Custom focal loss with class balancing
    "focal_gamma": 3.0,  # Increased gamma for harder examples
    "focal_alpha": 0.75,  # Weight for positive class
    "class_weight_ratio": 3.4,  # Decoy:Target ratio
    "model_size": "custom",  # Custom architecture
    "val_split": 0.1,  # 10% validation for better evaluation
    "seed": 42,
    "print_every": 50,
    "early_stopping_patience": 5,
    "label_smoothing": 0.05,  # Prevent overconfidence
    "mixup_alpha": 0.2,  # Data augmentation
    "dropout_increase": 0.1,  # Additional dropout
    "gradient_clip": 1.0,  # Gradient clipping
    "warmup_epochs": 2,  # Learning rate warmup
    "gpu_num": torch.cuda.device_count() if torch.cuda.is_available() else 1,
    "auto_score_after_training": True,  # Automatically score after training
    "samples_per_batch_scoring": 1000,  # For scoring
}

logger.info("=" * 80)
logger.info("Enhanced CNN Training with FDR Optimization and Integrated Scoring")
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
#                    Enhanced Data Loading with Augmentation                  #
# --------------------------------------------------------------------------- #

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def apply_mixup(rsm1, label1, rsm2, label2, alpha=0.2):
    """Apply mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    mixed_rsm = lam * rsm1 + (1 - lam) * rsm2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_rsm, mixed_label

def collate_batch_cnn_enhanced(batch_data):
    """Enhanced collate with data augmentation and balancing"""
    rsm_list = []
    label_list = []
    
    for batch in batch_data:
        rsm = batch["rsm"]
        # Process rsm: (72, 8, 16) → mean first 5 frag dims → (72, 16)
        if rsm.shape[1] >= 5:
            rsm_processed = rsm[:, :5, :].mean(axis=1)
        else:
            rsm_processed = rsm.mean(axis=1)
        
        # Add noise augmentation for training
        if np.random.random() < 0.3:  # 30% chance
            noise = np.random.normal(0, 0.01, rsm_processed.shape)
            rsm_processed = rsm_processed + noise
        
        rsm_list.append(rsm_processed)
        label_list.append(batch["label"])
    
    # Convert to tensors
    one_batch_rsm = torch.tensor(np.array(rsm_list), dtype=torch.float32)
    one_batch_label = torch.tensor(np.array(label_list), dtype=torch.float32)
    
    # Apply mixup augmentation
    if config["mixup_alpha"] > 0 and len(one_batch_rsm) > 1:
        indices = torch.randperm(len(one_batch_rsm))
        for i in range(0, len(one_batch_rsm) - 1, 2):
            if np.random.random() < 0.5:  # 50% chance to apply mixup
                mixed_rsm, mixed_label = apply_mixup(
                    one_batch_rsm[i].numpy(), one_batch_label[i].numpy(),
                    one_batch_rsm[indices[i]].numpy(), one_batch_label[indices[i]].numpy(),
                    config["mixup_alpha"]
                )
                one_batch_rsm[i] = torch.tensor(mixed_rsm, dtype=torch.float32)
                one_batch_label[i] = torch.tensor(mixed_label, dtype=torch.float32)
    
    # Add channel dimension for CNN
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
    """Custom dataset class for efficient data loading"""
    
    def __init__(self,
                 file_list: list,
                 file_bin_dict=None,
                 batch_size=1024,
                 buffer_size=2,
                 gpu_num=1,
                 shuffle=True,
                 seed=0,
                 multi_node=False,
                 is_training=True):
        super(IterableDiartDataset).__init__()
        self.file_list = file_list
        self.file_bin_dict = file_bin_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.gpu_num = gpu_num
        self.multi_node = multi_node
        self.buffer_size = buffer_size
        self.is_training = is_training
        
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
        
        collate_fn = collate_batch_cnn_enhanced if self.is_training else collate_batch_cnn_enhanced
        
        return DataLoader(data,
                          shuffle=False,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=0,
                          collate_fn=collate_fn)
    
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
    """Create iterable dataset for efficient data loading"""
    
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
            multi_node=multi_node,
            is_training=True
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
            multi_node=multi_node,
            is_training=False
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
            multi_node=multi_node,
            is_training=False
        )
        
        return val_dl

# --------------------------------------------------------------------------- #
#                    Enhanced Model Architecture                              #
# --------------------------------------------------------------------------- #

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EnhancedPeakCNN(nn.Module):
    """Enhanced CNN with residual connections and attention mechanisms"""
    def __init__(self, dropout_increase=0.1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Residual blocks with SE attention
        self.res_block1 = ResidualBlock(64, 128, stride=1)
        self.res_block2 = ResidualBlock(128, 256, stride=2)
        self.res_block3 = ResidualBlock(256, 512, stride=2)
        
        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced classifier with more regularization
        base_dropout = 0.5
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),  # *2 for concat of avg and max pool
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(base_dropout + dropout_increase),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(base_dropout + dropout_increase * 0.8),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(base_dropout + dropout_increase * 0.6),
            
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_logits=True):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Global pooling (both avg and max for better feature extraction)
        avg_pool = self.global_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        logits = self.classifier(x).squeeze(1)
        
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)

# --------------------------------------------------------------------------- #
#                       Enhanced Loss Functions                               #
# --------------------------------------------------------------------------- #

class FocalLossBalanced(nn.Module):
    """Focal loss with class balancing for imbalanced data"""
    def __init__(self, gamma=3.0, alpha=0.75, class_weight_ratio=3.4, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight_ratio = class_weight_ratio
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal weight
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply class weight for imbalance
        weight = torch.where(targets > 0.5, 
                            torch.tensor(self.class_weight_ratio).to(targets.device),
                            torch.tensor(1.0).to(targets.device))
        
        # Combine all weights
        loss = focal_weight * alpha_t * weight * bce_loss
        
        return loss.mean()

class RankingLoss(nn.Module):
    """Custom ranking loss to ensure targets score higher than decoys"""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, logits, targets):
        # Separate target and decoy predictions
        target_mask = targets > 0.5
        decoy_mask = ~target_mask
        
        if target_mask.sum() == 0 or decoy_mask.sum() == 0:
            return torch.tensor(0.0).to(logits.device)
        
        target_scores = logits[target_mask]
        decoy_scores = logits[decoy_mask]
        
        # Calculate pairwise ranking loss
        # We want: target_score > decoy_score + margin
        diff = target_scores.unsqueeze(1) - decoy_scores.unsqueeze(0) - self.margin
        ranking_loss = F.relu(-diff).mean()
        
        return ranking_loss

class CombinedLoss(nn.Module):
    """Combined loss for better FDR optimization"""
    def __init__(self, config):
        super().__init__()
        self.focal = FocalLossBalanced(
            gamma=config["focal_gamma"],
            alpha=config["focal_alpha"],
            class_weight_ratio=config["class_weight_ratio"],
            label_smoothing=config["label_smoothing"]
        )
        self.ranking = RankingLoss(margin=0.5)
        self.focal_weight = 0.7
        self.ranking_weight = 0.3
    
    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        ranking_loss = self.ranking(logits, targets)
        total_loss = self.focal_weight * focal_loss + self.ranking_weight * ranking_loss
        return total_loss

# --------------------------------------------------------------------------- #
#                    Learning Rate Scheduling                                 #
# --------------------------------------------------------------------------- #

class WarmupCosineScheduler:
    """Cosine scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# --------------------------------------------------------------------------- #
#                         Training utilities                                  #
# --------------------------------------------------------------------------- #

def calculate_fdr_metrics(logits, targets, threshold=0.5):
    """Calculate FDR and other metrics"""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # True/False Positives/Negatives
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        
        # Metrics
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 1  # Sensitivity/Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        
        # Score distributions
        target_scores = probs[targets > 0.5]
        decoy_scores = probs[targets < 0.5]
        
        if len(target_scores) > 0 and len(decoy_scores) > 0:
            mean_target_score = target_scores.mean().item()
            mean_decoy_score = decoy_scores.mean().item()
            score_separation = mean_target_score - mean_decoy_score
        else:
            mean_target_score = mean_decoy_score = score_separation = 0
        
        return {
            'fdr': fdr,
            'tpr': tpr,
            'precision': precision,
            'mean_target_score': mean_target_score,
            'mean_decoy_score': mean_decoy_score,
            'score_separation': score_separation
        }

def run_epoch_enhanced(dataset, model, optimizer, criterion, train: bool, epoch_num: int):
    if train:
        model.train()
        phase = "train"
    else:
        model.eval()
        phase = "val"
    
    t0 = time.time()
    total = correct = loss_sum = 0
    batch_count = 0
    all_metrics = defaultdict(list)
    
    logger.info(f"Starting {phase} epoch {epoch_num}...")
    
    for file_loader_idx, file_loader in enumerate(dataset):
        for batch_idx, (x, y) in enumerate(file_loader):
            if x.numel() == 0:
                continue
            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if train:
                optimizer.zero_grad(set_to_none=True)
            
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = criterion(logits, y)
                
                if train and config["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                
                if train:
                    loss.backward()
                    optimizer.step()
            
            # Calculate metrics
            metrics = calculate_fdr_metrics(logits, y)
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            preds = (logits.sigmoid() > .5).float()
            correct += (preds == y).sum().item()
            loss_sum += loss.item() * len(x)
            total += len(x)
            batch_count += 1
            
            if batch_count % config["print_every"] == 0:
                acc = correct / total if total > 0 else 0
                avg_loss = loss_sum / total if total > 0 else 0
                avg_fdr = np.mean(all_metrics['fdr'][-50:]) if all_metrics['fdr'] else 0
                avg_sep = np.mean(all_metrics['score_separation'][-50:]) if all_metrics['score_separation'] else 0
                
                logger.info(
                    f"  {phase} Epoch {epoch_num} | Batch {batch_count} | "
                    f"Loss {avg_loss:.4f} | Acc {acc:.4f} | "
                    f"FDR {avg_fdr:.3f} | Sep {avg_sep:.3f}"
                )
    
    if total == 0:
        logger.warning(f"No samples processed in {phase} epoch!")
        return 0.0, 0.0, {}
    
    # Aggregate metrics
    final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    final_metrics['accuracy'] = correct / total
    final_metrics['loss'] = loss_sum / total
    
    return loss_sum / total, correct / total, final_metrics

# --------------------------------------------------------------------------- #
#                    Early Stopping                                           #
# --------------------------------------------------------------------------- #

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score, model, path):
        score = val_score  # Higher is better for score separation
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0
    
    def save_checkpoint(self, model, path):
        torch.save(model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(), path)
        logger.info(f'Model saved to {path}')

# --------------------------------------------------------------------------- #
#                    Scoring Functions                                        #
# --------------------------------------------------------------------------- #

def prepare_windows_for_scoring(rsm_data, rt_values, samples_per_batch, window_size=16):
    """Prepare windows from RSM data for scoring"""
    aggregated = np.sum(rsm_data, axis=1)
    smoothed = gaussian_filter1d(aggregated, sigma=1, axis=-1)
    
    n_samples, n_mz, n_rt = smoothed.shape
    agg_factor = 3
    n_agg_rt = n_rt // agg_factor
    
    all_windows = []
    window_info = []
    
    for sample_idx in range(min(samples_per_batch, n_samples)):
        sample_rsm = smoothed[sample_idx]
        
        agg_rsm = np.zeros((n_mz, n_agg_rt))
        for i in range(n_agg_rt):
            start = i * agg_factor
            agg_rsm[:, i] = np.mean(sample_rsm[:, start:start+agg_factor], axis=1)
        
        max_windows = min(100, n_agg_rt - window_size + 1)
        for w_idx in range(max_windows):
            window = agg_rsm[:, w_idx:w_idx + window_size]
            w_min, w_max = window.min(), window.max()
            if w_max > w_min:
                window = (window - w_min) / (w_max - w_min)
            else:
                window = np.zeros_like(window)
            
            all_windows.append(window)
            window_info.append((sample_idx, w_idx))
    
    return np.array(all_windows) if all_windows else np.array([]), window_info

def process_single_batch_scoring(batch_num, folder_path, model, device):
    """Process a single batch and return results"""
    
    index_file = folder_path / f"batch_{batch_num}_index.txt"
    rsm_file = folder_path / f"batch_{batch_num}_rsm.npy"
    rt_file = folder_path / f"batch_{batch_num}_rt_values.npy"
    
    if not (index_file.exists() and rsm_file.exists() and rt_file.exists()):
        return None
    
    rsm_data = np.load(rsm_file)
    rt_values = np.load(rt_file)
    
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
    
    windows, window_info = prepare_windows_for_scoring(
        rsm_data, rt_values, config["samples_per_batch_scoring"]
    )
    
    if len(windows) == 0:
        return []
    
    windows = np.expand_dims(windows, axis=1)
    
    batch_size = 256
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            scores = model(batch, return_logits=False).cpu().numpy()
            all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    
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

def run_full_scoring(model, model_path, save_dir):
    """Run complete scoring on all NPY files"""
    
    logger.info("\n" + "="*70)
    logger.info("STARTING INTEGRATED SCORING PIPELINE")
    logger.info("="*70)
    
    # Create output directory for scoring results
    scoring_output_dir = save_dir / f'scoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    scoring_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Data folder: {SCORING_DATA_FOLDER}")
    logger.info(f"Output directory: {scoring_output_dir}")
    
    # Find maximum batch number
    folder_path = Path(SCORING_DATA_FOLDER)
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
    all_target_scores = []
    all_decoy_scores = []
    
    for batch_num in range(max_batch + 1):
        if batch_num % 10 == 0:
            logger.info(f"Processing batch {batch_num}...")
        
        results = process_single_batch_scoring(batch_num, folder_path, model, device)
        
        if results is not None:
            successful_batches += 1
            total_samples += len(results)
            
            # Collect scores for FDR calculation
            for r in results:
                if r['is_decoy']:
                    all_decoy_scores.append(r['score'])
                else:
                    all_target_scores.append(r['score'])
            
            results.sort(key=lambda x: x['score'], reverse=True)
            
            for rank, result in enumerate(results, 1):
                result['rank'] = rank
            
            n_targets = sum(1 for r in results if not r['is_decoy'])
            n_decoys = len(results) - n_targets
            
            output_data = {
                'batch_num': batch_num,
                'n_samples': len(results),
                'n_targets': n_targets,
                'n_decoys': n_decoys,
                'target_rate': n_targets / len(results) if len(results) > 0 else 0,
                'max_score': max(r['score'] for r in results) if results else 0,
                'min_score': min(r['score'] for r in results) if results else 0,
                'mean_score': sum(r['score'] for r in results) / len(results) if results else 0,
                'results': results
            }
            
            output_file = scoring_output_dir / f'batch_{batch_num:04d}_results.json'
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            if batch_num % 50 == 0 and batch_num > 0:
                logger.info(f"  Progress: {batch_num}/{max_batch} | Successful: {successful_batches} | Total samples: {total_samples}")
        else:
            failed_batches.append(batch_num)
    
    # Calculate global FDR at different thresholds
    logger.info("\nCalculating FDR at different score thresholds...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fdr_results = []
    
    for threshold in thresholds:
        n_target_above = sum(1 for s in all_target_scores if s >= threshold)
        n_decoy_above = sum(1 for s in all_decoy_scores if s >= threshold)
        total_above = n_target_above + n_decoy_above
        fdr = n_decoy_above / total_above if total_above > 0 else 0
        fdr_results.append({
            'threshold': threshold,
            'fdr': fdr,
            'n_targets': n_target_above,
            'n_decoys': n_decoy_above,
            'total': total_above
        })
        logger.info(f"  Threshold {threshold:.1f}: FDR={fdr:.3f}, Targets={n_target_above}, Decoys={n_decoy_above}")
    
    logger.info("-" * 50)
    logger.info("\nScoring complete!")
    logger.info(f"Successful batches: {successful_batches}")
    logger.info(f"Failed/missing batches: {len(failed_batches)}")
    logger.info(f"Total samples processed: {total_samples}")
    
    if all_target_scores and all_decoy_scores:
        logger.info(f"Mean target score: {np.mean(all_target_scores):.3f}")
        logger.info(f"Mean decoy score: {np.mean(all_decoy_scores):.3f}")
        logger.info(f"Score separation: {np.mean(all_target_scores) - np.mean(all_decoy_scores):.3f}")
    
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
        'mean_target_score': float(np.mean(all_target_scores)) if all_target_scores else 0,
        'mean_decoy_score': float(np.mean(all_decoy_scores)) if all_decoy_scores else 0,
        'score_separation': float(np.mean(all_target_scores) - np.mean(all_decoy_scores)) if all_target_scores and all_decoy_scores else 0,
        'fdr_analysis': fdr_results
    }
    
    summary_file = scoring_output_dir / '_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nAll results saved to directory: {scoring_output_dir}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("="*70)

# --------------------------------------------------------------------------- #
#                              MAIN PIPELINE                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ======================================================================= #
    #                          TRAINING PHASE                                 #
    # ======================================================================= #
    
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_iterable_dataset(
        TRAIN_DATA_PATHS, 
        config, 
        parse='train',
        multi_node=False
    )
    
    logger.info("Initializing enhanced model...")
    model = EnhancedPeakCNN(dropout_increase=config["dropout_increase"])
    
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel using GPUs: {list(range(NUM_GPUS))}")
    
    model = model.to(device)
    
    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 
         'lr': config["lr"] * 2},
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 
         'lr': config["lr"]}
    ], weight_decay=config["weight_decay"])
    
    scheduler = WarmupCosineScheduler(optimizer, config["warmup_epochs"], config["epochs"])
    criterion = CombinedLoss(config)
    
    # Save directory
    SAVE_DIR = Path(f"models_enhanced_fdr_{time.strftime('%Y%m%d_%H%M%S')}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {SAVE_DIR}")
    
    # Save configuration
    with open(SAVE_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting Enhanced Training with FDR Optimization")
    logger.info("=" * 80)
    
    best_val_sep = -float('inf')
    best_fdr = 1.0
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    
    training_history = {
        'train_loss': [], 'train_acc': [], 'train_fdr': [], 'train_sep': [],
        'val_loss': [], 'val_acc': [], 'val_fdr': [], 'val_sep': []
    }
    
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\n{'='*60}")
        current_lr = scheduler.step(epoch - 1)
        logger.info(f"Epoch {epoch}/{config['epochs']} -- LR {current_lr:.3e}")
        logger.info(f"{'='*60}")
        
        # Training
        tr_loss, tr_acc, tr_metrics = run_epoch_enhanced(
            train_dataset, model, optimizer, criterion, train=True, epoch_num=epoch
        )
        logger.info(f"Train: Loss={tr_loss:.4f}, Acc={tr_acc:.4f}, FDR={tr_metrics['fdr']:.3f}, "
                    f"Sep={tr_metrics['score_separation']:.3f}")
        
        # Validation
        val_loss, val_acc, val_metrics = run_epoch_enhanced(
            val_dataset, model, optimizer, criterion, train=False, epoch_num=epoch
        )
        logger.info(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, FDR={val_metrics['fdr']:.3f}, "
                    f"Sep={val_metrics['score_separation']:.3f}")
        
        # Update history
        for key, value in [('train_loss', tr_loss), ('train_acc', tr_acc), 
                          ('train_fdr', tr_metrics['fdr']), ('train_sep', tr_metrics['score_separation']),
                          ('val_loss', val_loss), ('val_acc', val_acc),
                          ('val_fdr', val_metrics['fdr']), ('val_sep', val_metrics['score_separation'])]:
            training_history[key].append(value)
        
        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "opt": optimizer.state_dict(),
            "metrics": val_metrics,
            "config": config,
        }
        
        ckpt_path = SAVE_DIR / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        
        # Check for best model
        if val_metrics['score_separation'] > best_val_sep and val_metrics['fdr'] < 0.3:
            best_val_sep = val_metrics['score_separation']
            best_fdr = val_metrics['fdr']
            best_path = SAVE_DIR / "best.pth"
            torch.save(ckpt, best_path)
            logger.info(f"✓ New best model! Sep={best_val_sep:.3f}, FDR={best_fdr:.3f}")
        
        # Early stopping
        early_stopping(val_metrics['score_separation'], model, SAVE_DIR / "early_stop_best.pth")
        if early_stopping.early_stop:
            logger.info("Early stopping triggered!")
            break
        
        # Save training history
        with open(SAVE_DIR / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"Training finished!")
    logger.info(f"Best score separation: {best_val_sep:.3f}")
    logger.info(f"Best FDR: {best_fdr:.3f}")
    logger.info(f"Models saved to: {SAVE_DIR}")
    logger.info("=" * 80)
    
    # ======================================================================= #
    #                          SCORING PHASE                                  #
    # ======================================================================= #
    
    if config["auto_score_after_training"]:
        logger.info("\n" + "="*80)
        logger.info("STARTING AUTOMATIC SCORING WITH BEST MODEL")
        logger.info("="*80)
        
        # Load the best model for scoring
        best_model_path = SAVE_DIR / "best.pth"
        if not best_model_path.exists():
            # If no best model (shouldn't happen), use the last epoch
            best_model_path = ckpt_path
            logger.warning("Best model not found, using last epoch model")
        
        # Load model for scoring
        scoring_model = EnhancedPeakCNN(dropout_increase=config["dropout_increase"]).to(device)
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'model' in checkpoint:
            scoring_model.load_state_dict(checkpoint['model'])
        else:
            scoring_model.load_state_dict(checkpoint)
        scoring_model.eval()
        
        logger.info(f"Loaded model from: {best_model_path}")
        
        # Run scoring
        run_full_scoring(scoring_model, best_model_path, SAVE_DIR)
        
        logger.info("\n" + "="*80)
        logger.info("COMPLETE PIPELINE FINISHED!")
        logger.info("Training and scoring results saved to: " + str(SAVE_DIR))
        logger.info("="*80)