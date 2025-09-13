import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

def separate_and_rank(results_dir):
    """Separate targets and decoys, rank them separately, and create visualizations"""
    
    print("="*70)
    print("SEPARATING AND RANKING TARGETS VS DECOYS")
    print("="*70)
    print(f"\nResults directory: {results_dir}\n")
    
    # Load all results
    print("Loading all results...")
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No results found!")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Separate targets and decoys
    targets_df = df[~df['is_decoy']].copy()
    decoys_df = df[df['is_decoy']].copy()
    
    # Sort and rank separately
    targets_df = targets_df.sort_values('score', ascending=False).reset_index(drop=True)
    decoys_df = decoys_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Add separate ranks
    targets_df['target_rank'] = range(1, len(targets_df) + 1)
    decoys_df['decoy_rank'] = range(1, len(decoys_df) + 1)
    
    # Print statistics
    print("-"*50)
    print("SEPARATION STATISTICS")
    print("-"*50)
    print(f"Total Targets: {len(targets_df):,}")
    print(f"Total Decoys: {len(decoys_df):,}")
    print(f"Target/Decoy Ratio: {len(targets_df)/len(decoys_df):.3f}")
    
    print(f"\nTarget Scores:")
    print(f"  Range: {targets_df['score'].min():.6f} - {targets_df['score'].max():.6f}")
    print(f"  Mean: {targets_df['score'].mean():.6f}")
    print(f"  Median: {targets_df['score'].median():.6f}")
    print(f"  Std: {targets_df['score'].std():.6f}")
    
    print(f"\nDecoy Scores:")
    print(f"  Range: {decoys_df['score'].min():.6f} - {decoys_df['score'].max():.6f}")
    print(f"  Mean: {decoys_df['score'].mean():.6f}")
    print(f"  Median: {decoys_df['score'].median():.6f}")
    print(f"  Std: {decoys_df['score'].std():.6f}")
    
    # Compare top performers
    print("\n" + "-"*50)
    print("TOP PERFORMERS COMPARISON")
    print("-"*50)
    
    top_n_values = [10, 100, 1000, 10000]
    for n in top_n_values:
        if n <= min(len(targets_df), len(decoys_df)):
            target_scores = targets_df.head(n)['score']
            decoy_scores = decoys_df.head(n)['score']
            
            print(f"\nTop {n:,}:")
            print(f"  Target score range: {target_scores.min():.6f} - {target_scores.max():.6f}")
            print(f"  Decoy score range: {decoy_scores.min():.6f} - {decoy_scores.max():.6f}")
            print(f"  Mean difference: {target_scores.mean() - decoy_scores.mean():.6f}")
    
    # Score overlap analysis
    print("\n" + "-"*50)
    print("SCORE OVERLAP ANALYSIS")
    print("-"*50)
    
    # Find score thresholds
    target_min, target_max = targets_df['score'].min(), targets_df['score'].max()
    decoy_min, decoy_max = decoys_df['score'].min(), decoys_df['score'].max()
    
    # Find overlap region
    overlap_min = max(target_min, decoy_min)
    overlap_max = min(target_max, decoy_max)
    
    if overlap_max > overlap_min:
        targets_in_overlap = ((targets_df['score'] >= overlap_min) & 
                             (targets_df['score'] <= overlap_max)).sum()
        decoys_in_overlap = ((decoys_df['score'] >= overlap_min) & 
                            (decoys_df['score'] <= overlap_max)).sum()
        
        print(f"Score overlap region: {overlap_min:.6f} - {overlap_max:.6f}")
        print(f"Targets in overlap: {targets_in_overlap:,} ({targets_in_overlap/len(targets_df)*100:.2f}%)")
        print(f"Decoys in overlap: {decoys_in_overlap:,} ({decoys_in_overlap/len(decoys_df)*100:.2f}%)")
    else:
        print("No score overlap between targets and decoys!")
    
    # Save separated results
    targets_file = os.path.join(results_dir, 'targets_ranked.csv')
    decoys_file = os.path.join(results_dir, 'decoys_ranked.csv')
    
    targets_df.to_csv(targets_file, index=False)
    decoys_df.to_csv(decoys_file, index=False)
    
    print(f"\n" + "="*70)
    print(f"Targets saved to: {targets_file}")
    print(f"Decoys saved to: {decoys_file}")
    
    return targets_df, decoys_df

def create_comparison_plots(targets_df, decoys_df, results_dir):
    """Create comprehensive comparison plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main plot: Score vs Rank (dotted lines)
    ax1 = plt.subplot(2, 3, 1)
    
    # Sample points for better visualization (log-spaced sampling)
    n_targets = len(targets_df)
    n_decoys = len(decoys_df)
    
    # Create log-spaced samples for smoother lines
    if n_targets > 1000:
        target_samples = np.unique(np.concatenate([
            np.arange(0, min(1000, n_targets)),  # First 1000
            np.logspace(np.log10(1000), np.log10(n_targets-1), 500).astype(int)
        ]))
    else:
        target_samples = np.arange(n_targets)
    
    if n_decoys > 1000:
        decoy_samples = np.unique(np.concatenate([
            np.arange(0, min(1000, n_decoys)),  # First 1000
            np.logspace(np.log10(1000), np.log10(n_decoys-1), 500).astype(int)
        ]))
    else:
        decoy_samples = np.arange(n_decoys)
    
    ax1.plot(target_samples + 1, targets_df.iloc[target_samples]['score'].values, 
             'b--', label='Targets', alpha=0.7, linewidth=1.5)
    ax1.plot(decoy_samples + 1, decoys_df.iloc[decoy_samples]['score'].values, 
             'r--', label='Decoys', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Score vs Rank Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Score vs Rank (linear scale for top performers)
    ax2 = plt.subplot(2, 3, 2)
    top_n = min(5000, n_targets, n_decoys)
    ax2.plot(range(1, top_n+1), targets_df.head(top_n)['score'].values, 
             'b--', label='Targets', alpha=0.7, linewidth=1.5)
    ax2.plot(range(1, top_n+1), decoys_df.head(top_n)['score'].values, 
             'r--', label='Decoys', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Rank', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Top {top_n:,} Scores (Linear Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Score distributions (histograms)
    ax3 = plt.subplot(2, 3, 3)
    bins = np.linspace(
        min(targets_df['score'].min(), decoys_df['score'].min()),
        max(targets_df['score'].max(), decoys_df['score'].max()),
        50
    )
    ax3.hist(targets_df['score'], bins=bins, alpha=0.5, label='Targets', 
             color='blue', density=True)
    ax3.hist(decoys_df['score'], bins=bins, alpha=0.5, label='Decoys', 
             color='red', density=True)
    ax3.set_xlabel('Score', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(np.sort(targets_df['score']), 
             np.linspace(0, 1, len(targets_df)), 
             'b-', label='Targets', linewidth=2)
    ax4.plot(np.sort(decoys_df['score']), 
             np.linspace(0, 1, len(decoys_df)), 
             'r-', label='Decoys', linewidth=2)
    ax4.set_xlabel('Score', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    ax5 = plt.subplot(2, 3, 5)
    box_data = [targets_df['score'].values, decoys_df['score'].values]
    bp = ax5.boxplot(box_data, labels=['Targets', 'Decoys'], 
                      patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.5)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Score Distribution Box Plot', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Score difference at each percentile
    ax6 = plt.subplot(2, 3, 6)
    percentiles = np.linspace(0, 100, 101)
    target_percentiles = np.percentile(targets_df['score'], percentiles)
    decoy_percentiles = np.percentile(decoys_df['score'], percentiles)
    score_diff = target_percentiles - decoy_percentiles
    
    ax6.plot(percentiles, score_diff, 'g-', linewidth=2)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.fill_between(percentiles, score_diff, 0, where=(score_diff > 0), 
                      alpha=0.3, color='green', label='Targets > Decoys')
    ax6.fill_between(percentiles, score_diff, 0, where=(score_diff < 0), 
                      alpha=0.3, color='red', label='Decoys > Targets')
    ax6.set_xlabel('Percentile', fontsize=12)
    ax6.set_ylabel('Score Difference (Target - Decoy)', fontsize=12)
    ax6.set_title('Score Difference by Percentile', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Target vs Decoy Score Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(results_dir, 'target_decoy_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_file}")
    plt.show()
    
    # Create a second figure for the simple comparison requested
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot with dotted lines as requested
    ax.plot(range(1, len(targets_df) + 1), targets_df['score'].values, 
            'b--', label='Targets', alpha=0.7, linewidth=1.5, marker='', markersize=0)
    ax.plot(range(1, len(decoys_df) + 1), decoys_df['score'].values, 
            'r--', label='Decoys', alpha=0.7, linewidth=1.5, marker='', markersize=0)
    
    ax.set_xlabel('Ranked Number', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Targets vs Decoys: Score by Rank', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Add some statistics as text
    textstr = f'Targets: {len(targets_df):,}\nDecoys: {len(decoys_df):,}\n' + \
              f'Target Mean: {targets_df["score"].mean():.4f}\n' + \
              f'Decoy Mean: {decoys_df["score"].mean():.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    simple_plot_file = os.path.join(results_dir, 'target_decoy_simple.png')
    plt.savefig(simple_plot_file, dpi=150, bbox_inches='tight')
    print(f"Simple comparison plot saved to: {simple_plot_file}")
    plt.show()

def main():
    # Specify your results directory
    RESULTS_DIR = "batch_results_20250912_180610"  # Change this to your actual directory
    
    # Separate and rank
    targets_df, decoys_df = separate_and_rank(RESULTS_DIR)
    
    if targets_df is not None and decoys_df is not None:
        # Create comparison plots
        create_comparison_plots(targets_df, decoys_df, RESULTS_DIR)
        
        # Additional analysis summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'n_targets': len(targets_df),
            'n_decoys': len(decoys_df),
            'target_score_stats': {
                'min': float(targets_df['score'].min()),
                'max': float(targets_df['score'].max()),
                'mean': float(targets_df['score'].mean()),
                'median': float(targets_df['score'].median()),
                'std': float(targets_df['score'].std())
            },
            'decoy_score_stats': {
                'min': float(decoys_df['score'].min()),
                'max': float(decoys_df['score'].max()),
                'mean': float(decoys_df['score'].mean()),
                'median': float(decoys_df['score'].median()),
                'std': float(decoys_df['score'].std())
            },
            'separation_quality': {
                'mean_difference': float(targets_df['score'].mean() - decoys_df['score'].mean()),
                'median_difference': float(targets_df['score'].median() - decoys_df['score'].median()),
                'best_target_score': float(targets_df['score'].max()),
                'best_decoy_score': float(decoys_df['score'].max()),
                'worst_target_score': float(targets_df['score'].min()),
                'worst_decoy_score': float(decoys_df['score'].min())
            }
        }
        
        summary_file = os.path.join(RESULTS_DIR, 'target_decoy_analysis.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nAnalysis summary saved to: {summary_file}")
        print("="*70)

if __name__ == "__main__":
    main()