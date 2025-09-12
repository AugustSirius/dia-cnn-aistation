import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime

# Configuration
MODEL_PATH = '/wangshuaiyao/jiangheng/dia-cnn/dia-cnn-aistation/model_epoch_004.pth'
print(f'load successful: {MODEL_PATH}')

DATA_FOLDER = '/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm/output_new'
print(f'load successful: {DATA_FOLDER}')

OUTPUT_FILE = f'scoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
MODEL_COMPLEXITY = "medium"
SAMPLES_PER_BATCH = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(OUTPUT_FILE, MODEL_COMPLEXITY, SAMPLES_PER_BATCH, device)

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

# MAIN EXECUTION - SUPER SIMPLE
print("="*70)
print("CNN MODEL - BATCH 0 ONLY")
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

# Process only batch_0
print("\nProcessing batch_0...")

# Define file paths directly
folder_path = Path(DATA_FOLDER)
index_file = folder_path / "batch_0_index.txt"
rsm_file = folder_path / "batch_0_rsm.npy"
rt_file = folder_path / "batch_0_rt_values.npy"

# Check files exist
if not index_file.exists():
    print(f"ERROR: {index_file} not found")
    exit(1)
if not rsm_file.exists():
    print(f"ERROR: {rsm_file} not found")
    exit(1)
if not rt_file.exists():
    print(f"ERROR: {rt_file} not found")
    exit(1)

print("Files found. Loading data...")

# Load data
rsm_data = np.load(rsm_file)
rt_values = np.load(rt_file)
print(f"RSM shape: {rsm_data.shape}")

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

print(f"Found {len(precursor_ids)} precursors")

# Process windows
print("Preparing windows...")
windows, window_info = prepare_windows(rsm_data, rt_values, SAMPLES_PER_BATCH)
print(f"Created {len(windows)} windows")

if len(windows) == 0:
    print("No windows created!")
    exit(1)

# Add channel dimension
windows = np.expand_dims(windows, axis=1)

# Score windows
print("Scoring windows...")
batch_size = 512
all_scores = []

with torch.no_grad():
    for i in range(0, len(windows), batch_size):
        batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
        scores = model(batch).cpu().numpy()
        all_scores.extend(scores)

all_scores = np.array(all_scores)
print(f"Scored {len(all_scores)} windows")

# Find best window for each sample
sample_results = {}
for (sample_idx, w_idx), score in zip(window_info, all_scores):
    if sample_idx not in sample_results or score > sample_results[sample_idx]['score']:
        sample_results[sample_idx] = {
            'score': float(score),
            'window_idx': int(w_idx),
            'is_decoy': is_decoy[sample_idx] if sample_idx < len(is_decoy) else False,
            'precursor_id': precursor_ids[sample_idx] if sample_idx < len(precursor_ids) else f"Sample_{sample_idx}",
            'batch_num': 0
        }

results = list(sample_results.values())
print(f"\nProcessed {len(results)} samples from batch_0")

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print top 10
print("\nTop 10 results:")
print("-" * 50)
for i, r in enumerate(results[:10]):
    print(f"{i+1}. Score: {r['score']:.4f} | {'DECOY' if r['is_decoy'] else 'TARGET'} | {r['precursor_id'][:40]}")

# Save results
output = {
    'batch': 0,
    'n_samples': len(results),
    'results': results[:100]  # Save only top 100 to keep file small
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")
print("="*70)