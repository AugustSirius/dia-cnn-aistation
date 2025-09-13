import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_all_results(results_dir):
    """Load all batch result JSON files from the directory"""
    all_results = []
    
    results_path = Path(results_dir)
    for json_file in sorted(results_path.glob("batch_*_results.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for result in data['results']:
                    all_results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_results

def plot_target_decoy_simple(results_dir):
    """Create a simple target vs decoy plot"""
    
    # Load all results
    print("Loading results...")
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Separate targets and decoys
    targets_df = df[~df['is_decoy']].copy()
    decoys_df = df[df['is_decoy']].copy()
    
    # Sort by score (ASCENDING - from low to high)
    targets_df = targets_df.sort_values('score', ascending=True).reset_index(drop=True)
    decoys_df = decoys_df.sort_values('score', ascending=True).reset_index(drop=True)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot targets and decoys
    plt.plot(range(len(targets_df)), targets_df['score'].values, 
             'b-', label='target', linewidth=2)
    plt.plot(range(len(decoys_df)), decoys_df['score'].values, 
             'r-', label='decoy', linewidth=2)
    
    plt.xlabel('Precursors', fontsize=12)
    plt.ylabel('score', fontsize=12)
    plt.title('cnn results', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(False)
    
    # Set y-axis limits
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(results_dir, 'target_decoy_plot.png')
    plt.savefig(output_file, dpi=150)
    plt.show()
    
    print(f"Plot saved to: {output_file}")
    print(f"Targets: {len(targets_df)}")
    print(f"Decoys: {len(decoys_df)}")

if __name__ == "__main__":
    # Specify your results directory
    RESULTS_DIR = "batch_results_20250912_180610"  # Change this to your actual directory
    
    plot_target_decoy_simple(RESULTS_DIR)