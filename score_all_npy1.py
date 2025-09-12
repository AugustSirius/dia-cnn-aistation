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