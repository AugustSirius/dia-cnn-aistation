#!/usr/bin/env python3
"""
Scoring script for CNN4-O3pro model
Compatible with the new training script architecture
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import re
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = '/Users/augustsirius/Desktop/dia-cnn-aistation/models_medium_20250912_225924/epoch_019.pth'
DATA_FOLDER = '/Users/augustsirius/Desktop/00.Project_DIA-CNN/dia-cnn/00_test_raw_input/test_scoring_dataset/'

OUTPUT_FILE = f'scoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
MODEL_COMPLEXITY = "medium"  # Must match the training configuration
SAMPLES_PER_BATCH = 1000  # Process all 1000 samples per batch

# Device selection
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

print(f"="*70)
print(f"CNN4-O3pro Model Scoring")
print(f"="*70)
print(f"Model Path: {MODEL_PATH}")
print(f"Data Folder: {DATA_FOLDER}")
print(f"Output File: {OUTPUT_FILE}")
print(f"Model Complexity: {MODEL_COMPLEXITY}")
print(f"Device: {device}")
print(f"="*70)

# ============================================================================
# Model Architecture (must match CNN4-O3pro.py exactly)
# ============================================================================

# Complexity presets (must match training)
PRESET = {
    "small": dict(ch=[16, 32, 64, 128], fc=[128, 64], drop=[0.3, 0.2]),
    "medium": dict(ch=[32, 64, 128, 256], fc=[256, 128], drop=[0.5, 0.3]),
    "large": dict(ch=[64, 128, 256, 512], fc=[512, 256], drop=[0.5, 0.3]),
    "xlarge": dict(ch=[128, 256, 512, 1024], fc=[1024, 512], drop=[0.5, 0.4]),
}[MODEL_COMPLEXITY]

class PeakCNN(nn.Module):
    """Model architecture matching CNN4-O3pro.py exactly"""
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
        return self.cls(self.conv(x)).squeeze(1)  # returns logits

# ============================================================================
# Helper Functions
# ============================================================================

def get_batch_files(data_folder):
    """Find all batch files in the folder"""
    folder_path = Path(data_folder)
    
    # Find all index files
    index_files = sorted(folder_path.glob("batch_*_index.txt"))
    
    batches = []
    for index_file in index_files:
        # Extract batch number
        match = re.search(r'batch_(\d+)_index', str(index_file))
        if match:
            batch_num = match.group(1)
            rsm_file = folder_path / f"batch_{batch_num}_rsm.npy"
            rt_file = folder_path / f"batch_{batch_num}_rt_values.npy"
            
            # Check if all files exist
            if rsm_file.exists() and rt_file.exists():
                batches.append({
                    'batch_num': int(batch_num),
                    'index_file': index_file,
                    'rsm_file': rsm_file,
                    'rt_file': rt_file
                })
    
    return sorted(batches, key=lambda x: x['batch_num'])

def prepare_windows_cnn4o3(rsm_data, rt_values=None, samples_per_batch=1000):
    """
    Prepare windows matching the CNN4-O3pro training data processing
    The training script processes RSM as: (72, 8, 16) → mean first 5 frag dims → (72, 16)
    """
    n_samples, n_channels, n_mz, n_rt = rsm_data.shape
    
    all_windows = []
    window_info = []
    
    for sample_idx in range(min(samples_per_batch, n_samples)):
        # Get sample RSM: (channels=8, mz=72, rt)
        sample_rsm = rsm_data[sample_idx]  # (8, 72, rt)
        
        # Average first 5 channels (matching training)
        if n_channels >= 5:
            rsm_processed = sample_rsm[:5, :, :].mean(axis=0)  # (72, rt)
        else:
            rsm_processed = sample_rsm.mean(axis=0)  # (72, rt)
        
        # Apply smoothing
        rsm_smoothed = gaussian_filter1d(rsm_processed, sigma=1, axis=-1)
        
        # Create sliding windows of size 16
        window_size = 16
        stride = 3  # You can adjust stride for overlap
        
        for w_idx in range(0, n_rt - window_size + 1, stride):
            window = rsm_smoothed[:, w_idx:w_idx + window_size]  # (72, 16)
            
            # Normalize window
            w_min, w_max = window.min(), window.max()
            if w_max > w_min:
                window = (window - w_min) / (w_max - w_min)
            else:
                window = np.zeros_like(window)
            
            all_windows.append(window)
            window_info.append((sample_idx, w_idx))
    
    return np.array(all_windows) if all_windows else np.array([]), window_info

def process_batch(batch_files, model, device, samples_per_batch=1000):
    """Process a single batch and return results"""
    
    # Load data
    rsm_data = np.load(batch_files['rsm_file'])  # Expected shape: (samples, channels, mz, rt)
    rt_values = np.load(batch_files['rt_file'])
    
    # Load metadata
    with open(batch_files['index_file'], 'r') as f:
        lines = f.readlines()
    
    precursor_ids = []
    is_decoy = []
    for line in lines:
        parts = line.strip().split('\t') if '\t' in line else line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            precursor_id = parts[1]
            precursor_ids.append(precursor_id)
            # Check for decoy patterns
            is_decoy.append(
                'DECOY' in precursor_id.upper() or 
                'decoy' in precursor_id.lower() or
                precursor_id.startswith('rev_') or
                precursor_id.startswith('REV_')
            )
    
    # Process windows using CNN4-O3pro compatible method
    windows, window_info = prepare_windows_cnn4o3(rsm_data, rt_values, samples_per_batch)
    
    if len(windows) == 0:
        print(f"Warning: No windows created for batch {batch_files['batch_num']}")
        return []
    
    # Add channel dimension: (N, 72, 16) -> (N, 1, 72, 16)
    windows = np.expand_dims(windows, axis=1)
    
    # Score windows
    batch_size = 256  # Smaller batch size for stability
    all_scores = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            # Model outputs logits, apply sigmoid for probabilities
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
                'batch_num': int(batch_files['batch_num'])
            }
    
    return list(sample_results.values())

def calculate_statistics(results, top_k):
    """Calculate target/decoy statistics for top K results"""
    if len(results) < top_k:
        top_k = len(results)
    
    top_results = results[:top_k]
    n_targets = sum(1 for r in top_results if not r['is_decoy'])
    n_decoys = top_k - n_targets
    
    return {
        'k': top_k,
        'targets': n_targets,
        'decoys': n_decoys,
        'target_rate': n_targets / top_k if top_k > 0 else 0
    }

# ============================================================================
# Main Processing
# ============================================================================

print("\nLoading model...")
model = PeakCNN().to(device)

# Load checkpoint (handle CNN4-O3pro format)
checkpoint = torch.load(MODEL_PATH, map_location=device)

# The CNN4-O3pro saves with this structure
if 'model' in checkpoint:
    # Handle DataParallel state dict if present
    state_dict = checkpoint['model']
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    if 'train_acc' in checkpoint:
        print(f"  Training accuracy: {checkpoint['train_acc']:.4f}")
else:
    # Try direct state dict loading
    model.load_state_dict(checkpoint)
    print("✓ Model state dict loaded directly")

model.eval()

# Find all batches
print(f"\nScanning folder: {DATA_FOLDER}")
batches = get_batch_files(DATA_FOLDER)
print(f"Found {len(batches)} batches to process")

if len(batches) == 0:
    print("ERROR: No valid batch files found!")
    print("Expected files pattern:")
    print("  - batch_*_index.txt")
    print("  - batch_*_rsm.npy")
    print("  - batch_*_rt_values.npy")
    exit(1)

# Process all batches
all_results = []
print("\nProcessing batches:")
print("-" * 50)

for i, batch_files in enumerate(batches):
    batch_num = batch_files['batch_num']
    progress = (i + 1) / len(batches) * 100
    print(f"[{i+1:3d}/{len(batches):3d}] Batch {batch_num:3d} ... ", end='', flush=True)
    
    try:
        batch_results = process_batch(batch_files, model, device, SAMPLES_PER_BATCH)
        all_results.extend(batch_results)
        print(f"✓ {len(batch_results):4d} samples | Progress: {progress:5.1f}%")
    except Exception as e:
        print(f"✗ Error: {e}")
        continue

print("-" * 50)
print(f"\nTotal samples processed: {len(all_results)}")

if len(all_results) == 0:
    print("ERROR: No samples were successfully processed!")
    exit(1)

# Sort results by score
all_results.sort(key=lambda x: x['score'], reverse=True)

# Add rank
for rank, result in enumerate(all_results, 1):
    result['rank'] = rank

# Calculate statistics
k_values = [100, 500, 1000, 2000, 5000, 10000, 100000]
statistics = []

print("\n" + "="*70)
print("PERFORMANCE STATISTICS")
print("="*70)
print(f"{'Top K':<10} {'Targets':<10} {'Decoys':<10} {'Target %':<12} {'Decoy %':<12}")
print("-"*70)

for k in k_values:
    if k <= len(all_results):
        stats = calculate_statistics(all_results, k)
        statistics.append(stats)
        print(f"{stats['k']:<10} {stats['targets']:<10} {stats['decoys']:<10} "
              f"{stats['target_rate']*100:<12.2f} {(1-stats['target_rate'])*100:<12.2f}")

# Print top 20
print("\n" + "="*70)
print("TOP 20 SCORING SAMPLES")
print("="*70)
print(f"{'Rank':<6} {'Score':<8} {'Type':<8} {'Batch':<8} {'Sample ID'}")
print("-"*70)

for result in all_results[:20]:
    sample_type = 'DECOY' if result['is_decoy'] else 'TARGET'
    sample_name = result['precursor_id']
    if len(sample_name) > 35:
        sample_name = sample_name[:32] + '...'
    print(f"{result['rank']:<6} {result['score']:<8.4f} {sample_type:<8} "
          f"{result['batch_num']:<8} {sample_name}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total batches processed: {len(batches)}")
print(f"Total samples scored: {len(all_results)}")
print(f"Score range: {min(r['score'] for r in all_results):.4f} - "
      f"{max(r['score'] for r in all_results):.4f}")
print(f"Mean score: {np.mean([r['score'] for r in all_results]):.4f}")
print(f"Median score: {np.median([r['score'] for r in all_results]):.4f}")

# Save results
output_data = {
    'metadata': {
        'model_path': MODEL_PATH,
        'data_folder': DATA_FOLDER,
        'model_complexity': MODEL_COMPLEXITY,
        'processing_date': datetime.now().isoformat(),
        'total_batches': len(batches),
        'total_samples': len(all_results),
        'device': str(device)
    },
    'statistics': statistics,
    'summary': {
        'score_range': {
            'min': float(min(r['score'] for r in all_results)),
            'max': float(max(r['score'] for r in all_results))
        },
        'mean_score': float(np.mean([r['score'] for r in all_results])),
        'median_score': float(np.median([r['score'] for r in all_results]))
    },
    'results': all_results
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n" + "="*70)
print(f"✓ Results saved to: {OUTPUT_FILE}")
print("="*70)