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


import torch
import sys
import platform

def test_pytorch_environment():
    """Test PyTorch installation and available computing devices"""
    
    print("=" * 60)
    print("PyTorch Environment Test")
    print("=" * 60)
    
    # System Information
    print("\n1. SYSTEM INFORMATION:")
    print(f"   Python Version: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # Check available devices
    print("\n2. AVAILABLE DEVICES:")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Number of CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Device Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Check MPS availability (Apple Silicon)
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    print(f"   MPS Available: {mps_available}")
    if mps_available:
        print(f"   MPS is built: {torch.backends.mps.is_built()}")
    
    # CPU is always available
    print(f"   CPU Available: True")
    print(f"   Number of CPU Threads: {torch.get_num_threads()}")
    
    # Determine best available device
    print("\n3. SELECTED DEVICE:")
    if cuda_available:
        device = torch.device("cuda")
        device_name = "CUDA GPU"
    elif mps_available:
        device = torch.device("mps")
        device_name = "Apple MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"   Using: {device_name} ({device})")
    
    # Perform simple tensor operations test
    print("\n4. TENSOR OPERATIONS TEST:")
    try:
        # Create random tensors
        print(f"   Creating tensors on {device_name}...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        print("   Performing matrix multiplication...")
        z = torch.matmul(x, y)
        
        # Check result
        print(f"   Result shape: {z.shape}")
        print(f"   Result device: {z.device}")
        print(f"   Mean value: {z.mean().item():.6f}")
        print("   ✓ Tensor operations successful!")
        
    except Exception as e:
        print(f"   ✗ Error during tensor operations: {e}")
        return False
    
    # Simple neural network test
    print("\n5. NEURAL NETWORK TEST:")
    try:
        # Create a simple model
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        print(f"   Creating simple neural network on {device_name}...")
        model = SimpleNet().to(device)
        
        # Create dummy input
        input_tensor = torch.randn(32, 10, device=device)
        
        # Forward pass
        print("   Performing forward pass...")
        output = model(input_tensor)
        
        # Backward pass
        print("   Performing backward pass...")
        loss = output.sum()
        loss.backward()
        
        print(f"   Output shape: {output.shape}")
        print(f"   Gradients computed: {model.fc1.weight.grad is not None}")
        print("   ✓ Neural network operations successful!")
        
    except Exception as e:
        print(f"   ✗ Error during neural network test: {e}")
        return False
    
    # Performance benchmark
    print("\n6. PERFORMANCE BENCHMARK:")
    try:
        import time
        
        # Matrix multiplication benchmark
        size = 2000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm-up
        for _ in range(3):
            _ = torch.matmul(a, b)
        
        # Synchronize if using CUDA
        if cuda_available and device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        
        if cuda_available and device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        print(f"   Matrix multiplication ({size}x{size}) x 10 iterations")
        print(f"   Total time: {elapsed_time:.3f} seconds")
        print(f"   Average time per operation: {elapsed_time/10:.3f} seconds")
        print("   ✓ Performance benchmark completed!")
        
    except Exception as e:
        print(f"   ✗ Error during benchmark: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_pytorch_environment()
        sys.exit(0 if success else 1)
    except ImportError:
        print("Error: PyTorch is not installed!")
        print("Install it using one of the following commands:")
        print("  - For CPU only: pip install torch")
        print("  - For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  - For MPS/Mac: pip install torch")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)