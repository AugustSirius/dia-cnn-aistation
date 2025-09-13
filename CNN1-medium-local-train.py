# ------------------------------------------------------------
# CNN-based scorer training for peak group classification
# ------------------------------------------------------------
import pickle, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from pathlib import Path
import os
from datetime import datetime

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ---- Configuration ----
TARGET_FOLDER = "/Users/augustsirius/Desktop/00.Project_DIA-CNN/dia-cnn/00_test_raw_input/test_training_dataset/targets"
DECOY_FOLDER = "/Users/augustsirius/Desktop/00.Project_DIA-CNN/dia-cnn/00_test_raw_input/test_training_dataset/decoys"
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Model complexity control - ADJUST THESE
MODEL_COMPLEXITY = "medium"  # Options: "small", "medium", "large", "xlarge"

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

# Create model save directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_DIR = f"models_{MODEL_COMPLEXITY}_{timestamp}"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ---- Dataset class definition (required for pickle loading) ----
class Dataset:
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
        return len(self.precursor_id)

    def fit_scale(self):
        pass

# ---- Data Loading ----
def load_and_process_folder(folder_path, label_value):
    folder = Path(folder_path)
    if not folder.exists():
        return np.array([]).reshape(0, 72, 16), np.array([])
    
    pkl_files = sorted(folder.glob("*.pkl"))
    X_list = []
    
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            ds = pickle.load(f)
            rsm = np.asarray(ds.rsm)  # (N, 72, 8, 16)
            X = rsm[..., :5, :].mean(axis=2)  # Average first 5 channels -> (N, 72, 16)
            X_list.append(X)
    
    if not X_list:
        return np.array([]).reshape(0, 72, 16), np.array([])
    
    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.full(len(X_combined), label_value, dtype=np.int64)
    return X_combined, y_combined

# Load data
print("Loading data...")
X_t, y_t = load_and_process_folder(TARGET_FOLDER, label_value=1)
X_d, y_d = load_and_process_folder(DECOY_FOLDER, label_value=0)

X = np.concatenate([X_t, X_d], axis=0).astype(np.float32)
y = np.concatenate([y_t, y_d], axis=0).astype(np.float32)

# Add channel dimension for CNN (N, 72, 16) -> (N, 1, 72, 16)
X = np.expand_dims(X, axis=1)

print(f"Dataset: {len(X)} samples, shape: {X.shape}")
print(f"Targets: {len(X_t)}, Decoys: {len(X_d)}")

# Split data
np.random.seed(42)
n_train = int(0.8 * len(X))
idx = np.random.permutation(len(X))
idx_train, idx_val = idx[:n_train], idx[n_train:]

X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]

print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

# ---- CNN Model Definition ----
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

# ---- Dataset and DataLoader ----
class PeakDataset(td.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.y)

train_loader = td.DataLoader(PeakDataset(X_train, y_train), 
                             batch_size=BATCH_SIZE, shuffle=True)
val_loader = td.DataLoader(PeakDataset(X_val, y_val), 
                           batch_size=BATCH_SIZE, shuffle=False)

# ---- Training ----
model = PeakGroupCNN(complexity=MODEL_COMPLEXITY).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Calculate total batches for progress tracking
total_train_batches = len(train_loader)
total_val_batches = len(val_loader)

print(f"\n{'='*60}")
print(f"Model Configuration: {MODEL_COMPLEXITY}")
print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Train batches: {total_train_batches}, Val batches: {total_val_batches}")
print(f"Saving models to: {MODEL_SAVE_DIR}")
print(f"{'='*60}\n")

best_val_acc = 0
best_epoch = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("Starting training...\n")

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    print("-" * 40)
    
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(xb)
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == yb).sum().item()
        train_total += len(yb)
        
        # Progress bar
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_train_batches:
            print(f"  Train: [{batch_idx+1}/{total_train_batches}] "
                  f"Loss: {train_loss/train_total:.4f} "
                  f"Acc: {train_correct/train_total:.4f}")
    
    train_loss /= train_total
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    print("  Validating...")
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(val_loader):
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            val_loss += loss.item() * len(xb)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == yb).sum().item()
            val_total += len(yb)
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    # Update learning rate
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Save model for this epoch
    model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, model_path)
    
    # Check if best model
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        best_epoch = epoch
        best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
    
    # Print epoch summary
    print(f"\n  Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"           Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"           LR: {current_lr:.6f} | Best: Epoch {best_epoch} ({best_val_acc:.4f})")
    if is_best:
        print("           *** NEW BEST MODEL ***")
    print()

# Save final model and training history
final_model_path = os.path.join(MODEL_SAVE_DIR, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)

history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history, f)

print(f"{'='*60}")
print(f"Training Complete!")
print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
print(f"Models saved to: {MODEL_SAVE_DIR}")
print(f"{'='*60}")