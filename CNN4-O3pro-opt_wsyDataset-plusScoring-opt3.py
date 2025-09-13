#!/usr/bin/env python3
# ------------------------------------------------------------
# Simplified CNN classifier with 1:1 balanced sampling
# No focal loss, no special techniques - just balanced training
# ------------------------------------------------------------

from __future__ import annotations
import os, pickle, random, time, math, json, logging, sys
from pathlib import Path
from collections import defaultdict
import glob
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
#                            Configuration                                    #
# --------------------------------------------------------------------------- #

# Data paths
TRAIN_DATA_PATHS = {
    "decoy": [
        "/guotiannan/train_data_20250830/decoy_base_reverse0",
        "/guotiannan/train_data_20250830/decoy_base_rt_shift"
    ],
    "target": [
        "/guotiannan/train_data_20250830/diann20_target",
        "/guotiannan/train_data_20250830/diann20_target_rt_shift"
    ]
}

# Scoring data path
SCORING_DATA_FOLDER = '/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm/output_new'

# Simple configuration
config = {
    "epochs": 20,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.1,  # 10% for validation
    "seed": 42,
    "print_every": 50,
    "samples_per_batch_scoring": 1000,
}

logger.info("=" * 80)
logger.info("Simplified CNN Training with 1:1 Balanced Sampling")
logger.info("=" * 80)
logger.info(f"Configuration: {config}")

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
#                          Device Setup                                       #
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# --------------------------------------------------------------------------- #
#                    Data Loading with 1:1 Balance                           #
# --------------------------------------------------------------------------- #

class PickleDataset(Dataset):
    """Dataset class for loading pickle files"""
    def __init__(self):
        self.rsm = None
        self.label = None
        self.file = None
        self.precursor_id = None

    def __getitem__(self, idx):
        return {
            "rsm": self.rsm[idx],
            "label": self.label[idx],
        }

    def __len__(self):
        return len(self.rsm) if self.rsm is not None else 0

def load_pickle_files(file_paths, label):
    """Load all pickle files and extract data"""
    all_rsm = []
    all_labels = []
    
    for file_path in file_paths:
        try:
            with open(file_path, "rb") as f:
                ds = pickle.load(f)
                if hasattr(ds, 'rsm') and ds.rsm is not None:
                    all_rsm.extend(ds.rsm)
                    all_labels.extend([label] * len(ds.rsm))
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue
    
    return all_rsm, all_labels

def create_balanced_dataset(data_paths, val_split=0.1):
    """Create balanced dataset with 1:1 decoy:target ratio"""
    
    logger.info("Loading data files...")
    
    # Load all decoy files
    decoy_rsm = []
    decoy_labels = []
    for decoy_path in data_paths["decoy"]:
        files = glob.glob(f'{decoy_path}/*.pkl')
        logger.info(f"Found {len(files)} files in {decoy_path}")
        rsm, labels = load_pickle_files(files, 0)  # Label 0 for decoy
        decoy_rsm.extend(rsm)
        decoy_labels.extend(labels)
    
    # Load all target files
    target_rsm = []
    target_labels = []
    for target_path in data_paths["target"]:
        files = glob.glob(f'{target_path}/*.pkl')
        logger.info(f"Found {len(files)} files in {target_path}")
        rsm, labels = load_pickle_files(files, 1)  # Label 1 for target
        target_rsm.extend(rsm)
        target_labels.extend(labels)
    
    logger.info(f"Loaded {len(decoy_rsm)} decoy samples")
    logger.info(f"Loaded {len(target_rsm)} target samples")
    
    # Balance the dataset - randomly sample decoys to match target count
    n_targets = len(target_rsm)
    if len(decoy_rsm) > n_targets:
        # Randomly sample decoys to match target count
        indices = np.random.choice(len(decoy_rsm), n_targets, replace=False)
        decoy_rsm = [decoy_rsm[i] for i in indices]
        decoy_labels = [decoy_labels[i] for i in indices]
        logger.info(f"Sampled {n_targets} decoys to match target count")
    
    # Combine and shuffle
    all_rsm = decoy_rsm + target_rsm
    all_labels = decoy_labels + target_labels
    
    # Shuffle together
    combined = list(zip(all_rsm, all_labels))
    random.shuffle(combined)
    all_rsm, all_labels = zip(*combined)
    
    # Convert to numpy arrays
    all_rsm = np.array(all_rsm)
    all_labels = np.array(all_labels)
    
    logger.info(f"Total balanced samples: {len(all_rsm)} (1:1 ratio)")
    
    # Split into train and validation
    n_val = int(len(all_rsm) * val_split)
    n_train = len(all_rsm) - n_val
    
    train_rsm = all_rsm[:n_train]
    train_labels = all_labels[:n_train]
    val_rsm = all_rsm[n_train:]
    val_labels = all_labels[n_train:]
    
    logger.info(f"Train samples: {n_train} ({(train_labels == 1).sum()} targets, {(train_labels == 0).sum()} decoys)")
    logger.info(f"Val samples: {n_val} ({(val_labels == 1).sum()} targets, {(val_labels == 0).sum()} decoys)")
    
    return (train_rsm, train_labels), (val_rsm, val_labels)

class SimpleDataset(Dataset):
    """Simple PyTorch dataset"""
    def __init__(self, rsm_data, labels):
        self.rsm = rsm_data
        self.labels = labels
    
    def __len__(self):
        return len(self.rsm)
    
    def __getitem__(self, idx):
        rsm = self.rsm[idx]
        
        # Process rsm: (72, 8, 16) → mean first 5 frag dims → (72, 16)
        if rsm.shape[1] >= 5:
            rsm_processed = rsm[:, :5, :].mean(axis=1)
        else:
            rsm_processed = rsm.mean(axis=1)
        
        # Convert to tensor and add channel dimension
        rsm_tensor = torch.tensor(rsm_processed, dtype=torch.float32).unsqueeze(0)  # (1, 72, 16)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return rsm_tensor, label_tensor

# --------------------------------------------------------------------------- #
#                       Simple CNN Model                                      #
# --------------------------------------------------------------------------- #

class SimpleCNN(nn.Module):
    """Simple CNN architecture without bells and whistles"""
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        logits = self.classifier(x).squeeze(1)
        return logits

# --------------------------------------------------------------------------- #
#                         Training Functions                                  #
# --------------------------------------------------------------------------- #

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = (torch.sigmoid(output) > 0.5).float()
        correct += (pred == target).sum().item()
        total += data.size(0)
        
        if batch_idx % config["print_every"] == 0:
            logger.info(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).float()
            correct += (pred == target).sum().item()
            total += data.size(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Calculate score separation
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    target_scores = all_probs[all_targets > 0.5]
    decoy_scores = all_probs[all_targets < 0.5]
    
    if len(target_scores) > 0 and len(decoy_scores) > 0:
        score_sep = target_scores.mean() - decoy_scores.mean()
    else:
        score_sep = 0
    
    return avg_loss, accuracy, score_sep

# --------------------------------------------------------------------------- #
#                        Scoring Functions                                    #
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
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            scores = torch.sigmoid(logits).cpu().numpy()
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
    logger.info("STARTING SCORING PIPELINE")
    logger.info("="*70)
    
    scoring_output_dir = save_dir / f'scoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    scoring_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Data folder: {SCORING_DATA_FOLDER}")
    logger.info(f"Output directory: {scoring_output_dir}")
    
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
                'results': results
            }
            
            output_file = scoring_output_dir / f'batch_{batch_num:04d}_results.json'
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            failed_batches.append(batch_num)
    
    # Calculate FDR at different thresholds
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
            'n_decoys': n_decoy_above
        })
        logger.info(f"  Threshold {threshold:.1f}: FDR={fdr:.3f}, Targets={n_target_above}, Decoys={n_decoy_above}")
    
    logger.info(f"\nScoring complete!")
    logger.info(f"Successful batches: {successful_batches}")
    logger.info(f"Total samples: {total_samples}")
    
    if all_target_scores and all_decoy_scores:
        logger.info(f"Mean target score: {np.mean(all_target_scores):.3f}")
        logger.info(f"Mean decoy score: {np.mean(all_decoy_scores):.3f}")
        logger.info(f"Score separation: {np.mean(all_target_scores) - np.mean(all_decoy_scores):.3f}")
    
    summary = {
        'model_path': str(model_path),
        'total_batches': max_batch + 1,
        'successful_batches': successful_batches,
        'total_samples': total_samples,
        'mean_target_score': float(np.mean(all_target_scores)) if all_target_scores else 0,
        'mean_decoy_score': float(np.mean(all_decoy_scores)) if all_decoy_scores else 0,
        'score_separation': float(np.mean(all_target_scores) - np.mean(all_decoy_scores)) if all_target_scores and all_decoy_scores else 0,
        'fdr_analysis': fdr_results
    }
    
    summary_file = scoring_output_dir / '_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {scoring_output_dir}")

# --------------------------------------------------------------------------- #
#                              MAIN PIPELINE                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    # ======================================================================= #
    #                          DATA LOADING                                   #
    # ======================================================================= #
    
    logger.info("Creating balanced dataset (1:1 ratio)...")
    (train_rsm, train_labels), (val_rsm, val_labels) = create_balanced_dataset(
        TRAIN_DATA_PATHS, 
        val_split=config["val_split"]
    )
    
    # Create PyTorch datasets and dataloaders
    train_dataset = SimpleDataset(train_rsm, train_labels)
    val_dataset = SimpleDataset(val_rsm, val_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ======================================================================= #
    #                          MODEL SETUP                                    #
    # ======================================================================= #
    
    logger.info("Initializing model...")
    model = SimpleCNN().to(device)
    
    # Simple binary cross entropy loss (no focal loss)
    criterion = nn.BCEWithLogitsLoss()
    
    # Simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # Save directory
    SAVE_DIR = Path(f"models_simple_balanced_{time.strftime('%Y%m%d_%H%M%S')}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {SAVE_DIR}")
    
    # Save configuration
    with open(SAVE_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # ======================================================================= #
    #                          TRAINING                                       #
    # ======================================================================= #
    
    logger.info("=" * 80)
    logger.info("Starting Training on Balanced Data")
    logger.info("=" * 80)
    
    best_val_acc = 0
    best_score_sep = 0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_sep': []
    }
    
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['epochs']} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_sep = validate_epoch(model, val_loader, criterion, device)
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Sep: {val_sep:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_sep'].append(val_sep)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_sep': val_sep
        }
        
        ckpt_path = SAVE_DIR / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
        
        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_sep > best_score_sep):
            best_val_acc = val_acc
            best_score_sep = val_sep
            best_path = SAVE_DIR / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model saved! Acc={val_acc:.4f}, Sep={val_sep:.4f}")
        
        # Save training history
        with open(SAVE_DIR / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best score separation: {best_score_sep:.4f}")
    logger.info("=" * 80)
    
    # ======================================================================= #
    #                          SCORING                                        #
    # ======================================================================= #
    
    logger.info("\n" + "="*80)
    logger.info("Starting Automatic Scoring with Best Model")
    logger.info("="*80)
    
    # Load best model for scoring
    best_model_path = SAVE_DIR / "best.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded best model from: {best_model_path}")
    
    # Run scoring
    run_full_scoring(model, best_model_path, SAVE_DIR)
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE PIPELINE FINISHED!")
    logger.info(f"All results saved to: {SAVE_DIR}")
    logger.info("="*80)