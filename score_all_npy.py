import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import re
import json
from datetime import datetime

# Configuration
# MODEL_PATH = '/Users/augustsirius/Desktop/00.Project_DIA-CNN/dia-cnn/20250912/models_medium_20250912_135513/model_epoch_004.pth'

MODEL_PATH = '/wangshuaiyao/jiangheng/dia-cnn/dia-cnn-aistation/model_epoch_004.pth'

print(f'load successful: {MODEL_PATH}')

# DATA_FOLDER = '/Users/augustsirius/Desktop/00.Project_DIA-CNN/dia-cnn/00_test_raw_input/test_scoring_dataset/'

DATA_FOLDER = '/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm/output_new'\

print(f'load successful: {DATA_FOLDER}')

# ------------------------------------------------------------------------------------------------

OUTPUT_FILE = f'scoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
MODEL_COMPLEXITY = "medium"  # Must match the training configuration
SAMPLES_PER_BATCH = 1000  # Process all 1000 samples per batch

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print(OUTPUT_FILE, MODEL_COMPLEXITY, SAMPLES_PER_BATCH, device)

# ------------------------------------------------------------------------------------------------

# Complexity presets (must match training)
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

# ---- Model Definition (must match training) ----
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

# ---- Helper Functions ----
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

def process_batch(batch_files, model, device, samples_per_batch=1000):
    """Process a single batch and return results"""
    
    # Load data
    rsm_data = np.load(batch_files['rsm_file'])
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
            scores = model(batch).cpu().numpy()
            all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    
    # Find best window for each sample
    sample_results = {}
    for (sample_idx, w_idx), score in zip(window_info, all_scores):
        if sample_idx not in sample_results or score > sample_results[sample_idx]['score']:
            sample_results[sample_idx] = {
                'score': float(score),  # Convert to float for JSON serialization
                'window_idx': int(w_idx),
                'is_decoy': is_decoy[sample_idx] if sample_idx < len(is_decoy) else False,
                'precursor_id': precursor_ids[sample_idx] if sample_idx < len(precursor_ids) else f"Sample_{sample_idx}",
                'batch_num': int(batch_files['batch_num'])
            }
    
    return list(sample_results.values())

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

# ---- Main Processing ----
print("="*70)
print("CNN MODEL BATCH PROCESSING")
print("="*70)

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
model = PeakGroupCNN(complexity=MODEL_COMPLEXITY).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model performance - Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}")
else:
    model.load_state_dict(checkpoint)
    print("Loaded model state dict")

model.eval()

# Find all batches
print(f"\nScanning folder: {DATA_FOLDER}")
batches = get_batch_files(DATA_FOLDER)
print(f"Found {len(batches)} batches to process")

# Process all batches with progress reporting
all_results = []
print("\nProcessing batches:")
print("-" * 50)

for i, batch_files in enumerate(batches):
    batch_num = batch_files['batch_num']
    progress = (i + 1) / len(batches) * 100
    print(f"[{i+1:3d}/{len(batches):3d}] Processing batch {batch_num:3d} ... ", end='')
    
    batch_results = process_batch(batch_files, model, device, SAMPLES_PER_BATCH)
    all_results.extend(batch_results)
    
    print(f"Done. {len(batch_results):4d} samples | Progress: {progress:5.1f}%")

print("-" * 50)
print(f"\nTotal samples processed: {len(all_results)}")

# Sort all results by score
all_results.sort(key=lambda x: x['score'], reverse=True)

# Add rank to each result
for rank, result in enumerate(all_results, 1):
    result['rank'] = rank

# ---- Calculate Statistics ----
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

# Calculate statistics for different K values
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

# ---- Print Top 20 Details ----
print("\n" + "="*70)
print("TOP 20 SCORING SAMPLES")
print("="*70)
print(f"{'Rank':<6} {'Score':<8} {'Type':<8} {'Batch':<8} {'Sample ID'}")
print("-"*70)

for rank, result in enumerate(all_results[:20], 1):
    sample_type = 'DECOY' if result['is_decoy'] else 'TARGET'
    sample_name = result['precursor_id']
    if len(sample_name) > 35:
        sample_name = sample_name[:32] + '...'
    print(f"{rank:<6} {result['score']:<8.4f} {sample_type:<8} {result['batch_num']:<8} {sample_name}")

# ---- Summary ----
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total batches processed: {len(batches)}")
print(f"Total samples scored: {len(all_results)}")
if all_results:
    print(f"Score range: {min(r['score'] for r in all_results):.4f} - {max(r['score'] for r in all_results):.4f}")
    print(f"Mean score: {np.mean([r['score'] for r in all_results]):.4f}")
    print(f"Median score: {np.median([r['score'] for r in all_results]):.4f}")

# ---- Save Results to File ----
output_data = {
    'metadata': {
        'model_path': MODEL_PATH,
        'data_folder': DATA_FOLDER,
        'model_complexity': MODEL_COMPLEXITY,
        'processing_date': datetime.now().isoformat(),
        'total_batches': len(batches),
        'total_samples': len(all_results),
        'device': device
    },
    'statistics': statistics,
    'summary': {
        'score_range': {
            'min': float(min(r['score'] for r in all_results)) if all_results else 0,
            'max': float(max(r['score'] for r in all_results)) if all_results else 0
        },
        'mean_score': float(np.mean([r['score'] for r in all_results])) if all_results else 0,
        'median_score': float(np.median([r['score'] for r in all_results])) if all_results else 0
    },
    'results': all_results
}

# Save to JSON file
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n" + "="*70)
print(f"Results saved to: {OUTPUT_FILE}")
print("="*70)