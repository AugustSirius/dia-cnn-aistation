import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd

def load_all_results(results_dir):
    """Load all batch result JSON files from the directory"""
    all_results = []
    batch_files = []
    
    # Find all batch result files
    results_path = Path(results_dir)
    for json_file in sorted(results_path.glob("batch_*_results.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Extract individual results from each batch
                for result in data['results']:
                    # Add batch info if not present
                    if 'batch_num' not in result:
                        result['batch_num'] = data['batch_num']
                    all_results.append(result)
                batch_files.append(json_file.name)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(batch_files)} batch files")
    return all_results

def analyze_results(results_dir):
    """Main analysis function"""
    print("="*70)
    print("ANALYZING CNN SCORING RESULTS")
    print("="*70)
    print(f"\nResults directory: {results_dir}\n")
    
    # Load all results
    print("Loading all results...")
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Sort by score (descending)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Add global rank
    df['global_rank'] = range(1, len(df) + 1)
    
    # Basic statistics
    total_samples = len(df)
    total_targets = (~df['is_decoy']).sum()
    total_decoys = df['is_decoy'].sum()
    
    print("-"*50)
    print("OVERALL STATISTICS")
    print("-"*50)
    print(f"Total samples: {total_samples:,}")
    print(f"Total targets: {total_targets:,} ({total_targets/total_samples*100:.2f}%)")
    print(f"Total decoys: {total_decoys:,} ({total_decoys/total_samples*100:.2f}%)")
    print(f"\nScore range: {df['score'].min():.6f} - {df['score'].max():.6f}")
    print(f"Mean score: {df['score'].mean():.6f}")
    print(f"Median score: {df['score'].median():.6f}")
    
    # Analyze top N scores
    print("\n" + "-"*50)
    print("TOP SCORES ANALYSIS")
    print("-"*50)
    
    top_n_values = [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    
    analysis_results = []
    for n in top_n_values:
        if n <= total_samples:
            top_n_df = df.head(n)
            n_targets = (~top_n_df['is_decoy']).sum()
            n_decoys = top_n_df['is_decoy'].sum()
            target_rate = n_targets / n * 100
            min_score = top_n_df['score'].min()
            max_score = top_n_df['score'].max()
            
            print(f"\nTop {n:,} scores:")
            print(f"  Targets: {n_targets:,} ({target_rate:.2f}%)")
            print(f"  Decoys: {n_decoys:,} ({100-target_rate:.2f}%)")
            print(f"  Score range: {min_score:.6f} - {max_score:.6f}")
            print(f"  Target/Decoy ratio: {n_targets/n_decoys:.3f}" if n_decoys > 0 else "  Target/Decoy ratio: inf")
            
            analysis_results.append({
                'top_n': n,
                'n_targets': n_targets,
                'n_decoys': n_decoys,
                'target_rate': target_rate,
                'min_score': min_score,
                'max_score': max_score
            })
    
    # Score distribution by decile
    print("\n" + "-"*50)
    print("SCORE DISTRIBUTION BY DECILE")
    print("-"*50)
    
    df['decile'] = pd.qcut(df['global_rank'], 10, labels=False) + 1
    
    for decile in range(1, 11):
        decile_df = df[df['decile'] == decile]
        n_targets = (~decile_df['is_decoy']).sum()
        n_decoys = decile_df['is_decoy'].sum()
        target_rate = n_targets / len(decile_df) * 100
        score_range = f"{decile_df['score'].min():.4f}-{decile_df['score'].max():.4f}"
        
        print(f"Decile {decile:2d}: Targets: {n_targets:6,} ({target_rate:5.1f}%) | "
              f"Decoys: {n_decoys:6,} | Score: {score_range}")
    
    # Find score thresholds for different FDR levels
    print("\n" + "-"*50)
    print("FDR ANALYSIS (False Discovery Rate)")
    print("-"*50)
    
    fdr_levels = [0.01, 0.05, 0.10, 0.20]
    
    for target_fdr in fdr_levels:
        # Find the score threshold that achieves this FDR
        for i in range(len(df)):
            subset = df.head(i+1)
            n_decoys = subset['is_decoy'].sum()
            n_total = len(subset)
            current_fdr = n_decoys / n_total if n_total > 0 else 0
            
            if current_fdr > target_fdr:
                if i > 0:
                    # Use previous threshold
                    threshold_idx = i - 1
                    threshold_score = df.iloc[threshold_idx]['score']
                    n_passing = threshold_idx + 1
                    n_targets = (~df.head(n_passing)['is_decoy']).sum()
                    n_decoys = df.head(n_passing)['is_decoy'].sum()
                    actual_fdr = n_decoys / n_passing
                    
                    print(f"\nFDR ≤ {target_fdr*100:.0f}%:")
                    print(f"  Score threshold: {threshold_score:.6f}")
                    print(f"  Samples passing: {n_passing:,}")
                    print(f"  Targets: {n_targets:,}")
                    print(f"  Decoys: {n_decoys:,}")
                    print(f"  Actual FDR: {actual_fdr*100:.2f}%")
                break
    
    # Save detailed results
    output_file = os.path.join(results_dir, 'aggregate_analysis.json')
    
    analysis_summary = {
        'analysis_date': datetime.now().isoformat(),
        'results_directory': str(results_dir),
        'total_samples': int(total_samples),
        'total_targets': int(total_targets),
        'total_decoys': int(total_decoys),
        'overall_target_rate': float(total_targets/total_samples),
        'score_statistics': {
            'min': float(df['score'].min()),
            'max': float(df['score'].max()),
            'mean': float(df['score'].mean()),
            'median': float(df['score'].median()),
            'std': float(df['score'].std())
        },
        'top_n_analysis': analysis_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\n" + "="*70)
    print(f"Analysis summary saved to: {output_file}")
    
    # Save sorted results to CSV for further analysis
    csv_file = os.path.join(results_dir, 'all_results_sorted.csv')
    df.to_csv(csv_file, index=False)
    print(f"Sorted results saved to: {csv_file}")
    
    # Optional: Save top performers
    top_performers_file = os.path.join(results_dir, 'top_1000_results.csv')
    df.head(1000).to_csv(top_performers_file, index=False)
    print(f"Top 1000 results saved to: {top_performers_file}")
    
    print("="*70)
    
    return df, analysis_summary

def plot_results(df, results_dir):
    """Optional: Create visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Score distribution
        ax = axes[0, 0]
        ax.hist([df[df['is_decoy']]['score'], df[~df['is_decoy']]['score']], 
                bins=50, label=['Decoys', 'Targets'], alpha=0.7)
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.legend()
        
        # 2. Cumulative target rate
        ax = axes[0, 1]
        cumulative_targets = (~df['is_decoy']).cumsum()
        cumulative_rate = cumulative_targets / (df.index + 1)
        ax.plot(df.index[:10000], cumulative_rate[:10000])
        ax.set_xlabel('Rank (top N)')
        ax.set_ylabel('Target Rate')
        ax.set_title('Cumulative Target Rate (Top 10,000)')
        ax.grid(True, alpha=0.3)
        
        # 3. ROC-like curve
        ax = axes[1, 0]
        n_targets_cumsum = (~df['is_decoy']).cumsum()
        n_decoys_cumsum = df['is_decoy'].cumsum()
        tpr = n_targets_cumsum / total_targets
        fpr = n_decoys_cumsum / total_decoys
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-like Curve')
        ax.grid(True, alpha=0.3)
        
        # 4. Score vs Rank
        ax = axes[1, 1]
        sample_indices = np.logspace(0, np.log10(len(df)-1), 1000).astype(int)
        ax.plot(sample_indices, df.iloc[sample_indices]['score'].values)
        ax.set_xlabel('Rank (log scale)')
        ax.set_ylabel('Score')
        ax.set_title('Score vs Rank')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(results_dir, 'analysis_plots.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {plot_file}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")

if __name__ == "__main__":
    # Specify your results directory
    RESULTS_DIR = "batch_results_20250912_180610"  # Change this to your actual directory
    
    # Run analysis
    df, summary = analyze_results(RESULTS_DIR)
    
    # Optional: Create plots if matplotlib is available
    if df is not None:
        plot_results(df, RESULTS_DIR)