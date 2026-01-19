#!/usr/bin/env python3
"""
100 Epochs Augmented Test Experiment - Comprehensive Comparison
================================================================

Compares the new 100ep_lr0001_halfdata_augtest experiment with:
1. Previous augmented test experiments (30ep, 50ep)
2. Regular half data experiments (no augmented test)
3. Full data experiments

Generates comprehensive plots and analysis.

Output: testing/100ep_lr0001_halfdata_augtest/comparison_plots/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


# ============================================
# Configuration
# ============================================
MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]
MODEL_COLORS = {"ResNet18": "#e41a1c", "EfficientNet": "#377eb8", "ConvNeXt": "#4daf4a"}
MODEL_MARKERS = {"ResNet18": "o", "EfficientNet": "s", "ConvNeXt": "^"}

# All experiments to compare
EXPERIMENTS = {
    # Augmented test experiments
    "30ep Aug": {
        "path": Path("testing/30ep_lr0001_halfdata_augtest"),
        "color": "#9467bd",
        "epochs": 30,
        "type": "augmented"
    },
    "50ep Aug": {
        "path": Path("testing/50ep_lr0001_halfdata_augtest"),
        "color": "#bcbd22",
        "epochs": 50,
        "type": "augmented"
    },
    "100ep Aug": {
        "path": Path("testing/100ep_lr0001_halfdata_augtest"),
        "color": "#d62728",
        "epochs": 100,
        "type": "augmented"
    },
    # Regular half data (for reference)
    "30ep Half": {
        "path": Path("testing/30epochs_lr0.0001_halfdata"),
        "color": "#2ca02c",
        "epochs": 30,
        "type": "regular"
    },
    "50ep Half": {
        "path": Path("testing/50ep_lr0001_halfdata"),
        "color": "#17becf",
        "epochs": 50,
        "type": "regular"
    },
    "100ep Half": {
        "path": Path("testing/100ep_lr0001_halfdata"),
        "color": "#ff7f0e",
        "epochs": 100,
        "type": "regular"
    },
}

OUTPUT_DIR = Path("testing/100ep_lr0001_halfdata_augtest/comparison_plots")


def load_experiment_results(exp_path):
    """Load results for all models in an experiment."""
    results = {}
    
    for model in MODELS:
        model_dir = exp_path / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
            continue
        
        errors = np.load(errors_path)
        
        results_file = model_dir / "test_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                test_results = json.load(f)
        else:
            test_results = {}
        
        history_file = model_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {}
        
        results[model] = {
            "errors": errors,
            "history": history,
            **test_results
        }
    
    return results


def plot_augmented_comparison():
    """Compare all augmented test experiments."""
    print("\n📊 Generating Augmented Test Experiments Comparison...")
    
    aug_experiments = {k: v for k, v in EXPERIMENTS.items() if v["type"] == "augmented"}
    
    # Load data
    all_results = {}
    for exp_name, config in aug_experiments.items():
        if config["path"].exists():
            all_results[exp_name] = load_experiment_results(config["path"])
    
    if not all_results:
        print("  ⚠ No augmented experiments found!")
        return
    
    # 1. Mean Error Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    exp_names = list(all_results.keys())
    x = np.arange(len(exp_names))
    width = 0.25
    
    # Mean error
    ax = axes[0]
    for i, model in enumerate(MODELS):
        means = []
        for exp in exp_names:
            if model in all_results[exp]:
                means.append(all_results[exp][model].get('mean_error', 0))
            else:
                means.append(0)
        ax.bar(x + i*width, means, width, label=model, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Mean Error by Experiment (Augmented Test)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(exp_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Within 10m accuracy
    ax = axes[1]
    for i, model in enumerate(MODELS):
        acc = []
        for exp in exp_names:
            if model in all_results[exp]:
                acc.append(all_results[exp][model].get('within_10m', 0))
            else:
                acc.append(0)
        ax.bar(x + i*width, acc, width, label=model, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Within 10m Accuracy (Augmented Test)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(exp_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_augmented_comparison_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_augmented_comparison_bars.png")
    
    # 2. Error Distribution Box Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        data = []
        labels = []
        for exp in exp_names:
            if model in all_results[exp]:
                data.append(all_results[exp][model]['errors'])
                labels.append(exp)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(0.6)
        
        ax.set_ylabel('Error (m)')
        ax.set_title(f'{model} Error Distribution')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Error Distribution Across Augmented Test Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_augmented_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_augmented_boxplots.png")
    
    return all_results


def plot_augmented_vs_regular():
    """Compare augmented vs regular test experiments."""
    print("\n📊 Generating Augmented vs Regular Comparison...")
    
    # Load all experiments
    all_results = {}
    for exp_name, config in EXPERIMENTS.items():
        if config["path"].exists():
            all_results[exp_name] = load_experiment_results(config["path"])
    
    # Focus on 100 epoch comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    exp_100 = ["100ep Half", "100ep Aug"]
    available = [e for e in exp_100 if e in all_results]
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        
        data = []
        labels = []
        colors = []
        for exp in available:
            if model in all_results[exp]:
                data.append(all_results[exp][model]['errors'])
                labels.append(exp.replace(" ", "\n"))
                colors.append(EXPERIMENTS[exp]["color"])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Error (m)')
        ax.set_title(f'{model}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('100 Epochs: Regular Test vs Augmented Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_100ep_regular_vs_augmented.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_100ep_regular_vs_augmented.png")


def plot_training_curves():
    """Plot training curves for 100ep augmented experiment."""
    print("\n📊 Generating Training Curves...")
    
    exp_path = EXPERIMENTS["100ep Aug"]["path"]
    results = load_experiment_results(exp_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        
        if model in results and 'history' in results[model]:
            history = results[model]['history']
            
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], '-', 
                       color=MODEL_COLORS[model], linewidth=2, label='Validation Loss (m)')
            
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax2 = ax.twinx()
                ax2.plot(epochs, history['train_loss'], '--', 
                        color=MODEL_COLORS[model], alpha=0.5, linewidth=1, label='Train Loss')
                ax2.set_ylabel('Train Loss (L1)', color='gray')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss (m)')
        ax.set_title(f'{model}')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.suptitle('Training Curves: 100 Epochs, LR=0.0001, Half Data, Augmented Test', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_training_curves.png")


def plot_cdf_comparison():
    """Plot CDF of errors for all models in 100ep augmented experiment."""
    print("\n📊 Generating CDF Plot...")
    
    exp_path = EXPERIMENTS["100ep Aug"]["path"]
    results = load_experiment_results(exp_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in MODELS:
        if model in results:
            errors = np.sort(results[model]['errors'])
            cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
            ax.plot(errors, cdf, '-', color=MODEL_COLORS[model], 
                   linewidth=2, label=model)
    
    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90%')
    ax.axvline(x=10, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=20, color='black', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Error (meters)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('CDF of Prediction Errors\n100 Epochs, LR=0.0001, Half Data, Augmented Test')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_cdf_errors.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_cdf_errors.png")


def plot_epoch_progression():
    """Show how accuracy improves with more epochs (augmented tests only)."""
    print("\n📊 Generating Epoch Progression Plot...")
    
    aug_experiments = {k: v for k, v in EXPERIMENTS.items() if v["type"] == "augmented"}
    
    all_results = {}
    for exp_name, config in aug_experiments.items():
        if config["path"].exists():
            all_results[exp_name] = load_experiment_results(config["path"])
    
    epochs = [30, 50, 100]
    exp_order = ["30ep Aug", "50ep Aug", "100ep Aug"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean Error vs Epochs
    ax = axes[0]
    for model in MODELS:
        mean_errors = []
        valid_epochs = []
        for exp, ep in zip(exp_order, epochs):
            if exp in all_results and model in all_results[exp]:
                mean_errors.append(all_results[exp][model].get('mean_error', np.nan))
                valid_epochs.append(ep)
        
        if mean_errors:
            ax.plot(valid_epochs, mean_errors, '-o', 
                   color=MODEL_COLORS[model], linewidth=2, 
                   markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Mean Error vs Training Epochs')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Within 10m vs Epochs
    ax = axes[1]
    for model in MODELS:
        accuracy = []
        valid_epochs = []
        for exp, ep in zip(exp_order, epochs):
            if exp in all_results and model in all_results[exp]:
                accuracy.append(all_results[exp][model].get('within_10m', np.nan))
                valid_epochs.append(ep)
        
        if accuracy:
            ax.plot(valid_epochs, accuracy, '-o', 
                   color=MODEL_COLORS[model], linewidth=2, 
                   markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Within 10m (%)')
    ax.set_title('Accuracy vs Training Epochs')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Performance Progression (Augmented Test)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_epoch_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_epoch_progression.png")


def create_summary_table():
    """Create a comprehensive summary table."""
    print("\n📊 Generating Summary Table...")
    
    # Load 100ep augmented results
    exp_path = EXPERIMENTS["100ep Aug"]["path"]
    results = load_experiment_results(exp_path)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ['Model', 'Mean Error', 'Median Error', 'Std Dev', 
               '<5m', '<10m', '<15m', '<20m', '<50m', 'P90', 'P95']
    rows = []
    
    for model in MODELS:
        if model in results:
            r = results[model]
            row = [
                model,
                f"{r.get('mean_error', 0):.2f}m",
                f"{r.get('median_error', 0):.2f}m",
                f"{r.get('std_error', 0):.2f}m",
                f"{r.get('within_5m', 0):.1f}%",
                f"{r.get('within_10m', 0):.1f}%",
                f"{r.get('within_15m', 0):.1f}%",
                f"{r.get('within_20m', 0):.1f}%",
                f"{r.get('within_50m', 0):.1f}%",
                f"{r.get('p90_error', 0):.2f}m",
                f"{r.get('p95_error', 0):.2f}m",
            ]
            rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Model column styling
    for i, model in enumerate(MODELS):
        if i < len(rows):
            table[(i+1, 0)].set_facecolor(MODEL_COLORS[model])
            table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best values
    for col in range(1, len(columns)):
        values = []
        for row in range(len(rows)):
            cell_text = rows[row][col]
            try:
                val = float(cell_text.replace('m', '').replace('%', ''))
                values.append((val, row))
            except:
                pass
        
        if values:
            # For accuracy columns (%), higher is better
            if '%' in rows[0][col]:
                best_idx = max(values, key=lambda x: x[0])[1]
            else:
                # For error columns, lower is better
                best_idx = min(values, key=lambda x: x[0])[1]
            
            table[(best_idx + 1, col)].set_facecolor('#90EE90')
    
    plt.title("100 Epochs, LR=0.0001, Half Data, Augmented Test\nDetailed Results Summary", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_summary_table.png")


def plot_error_histogram():
    """Plot error histogram for each model."""
    print("\n📊 Generating Error Histograms...")
    
    exp_path = EXPERIMENTS["100ep Aug"]["path"]
    results = load_experiment_results(exp_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        
        if model in results:
            errors = results[model]['errors']
            
            ax.hist(errors, bins=30, color=MODEL_COLORS[model], 
                   alpha=0.7, edgecolor='black')
            
            mean_err = np.mean(errors)
            median_err = np.median(errors)
            
            ax.axvline(x=mean_err, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_err:.1f}m')
            ax.axvline(x=median_err, color='blue', linestyle=':', 
                      linewidth=2, label=f'Median: {median_err:.1f}m')
            
            ax.set_xlabel('Error (m)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{model}')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.suptitle('Error Distribution: 100 Epochs, Augmented Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_error_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_error_histograms.png")


def main():
    """Generate all comparison plots."""
    print("=" * 70)
    print("100 EPOCHS AUGMENTED TEST - COMPREHENSIVE COMPARISON")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_augmented_comparison()
    plot_augmented_vs_regular()
    plot_training_curves()
    plot_cdf_comparison()
    plot_epoch_progression()
    create_summary_table()
    plot_error_histogram()
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Generated 8 comparison plots")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
