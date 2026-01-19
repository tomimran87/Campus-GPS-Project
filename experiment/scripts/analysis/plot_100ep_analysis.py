#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for 100 epoch experiments.
Covers both lr=0.001 and lr=0.0001 full data experiments.

This is our BEST performing configuration:
- EfficientNet 100ep lr=0.001 fulldata: 5.71m mean error, 86.9% within 10m
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


# Configuration
EXPERIMENTS = {
    "100ep_lr001_fulldata": {
        "path": Path("testing/100ep_lr001_fulldata"),
        "label": "100ep LR=0.001",
        "color": "#e41a1c",
    },
    "100ep_lr0001_fulldata": {
        "path": Path("testing/100ep_lr0001_fulldata"),
        "label": "100ep LR=0.0001",
        "color": "#377eb8",
    },
}

MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]
MODEL_COLORS = {"ResNet18": "#e41a1c", "EfficientNet": "#377eb8", "ConvNeXt": "#4daf4a"}

OUTPUT_DIR = Path("testing/analysis_100epochs")


def load_experiment_results(exp_path):
    """Load results for all models in an experiment."""
    results = {}
    histories = {}
    
    for model in MODELS:
        model_dir = exp_path / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
            print(f"  ⚠ {model} not found in {exp_path}")
            continue
        
        errors = np.load(errors_path)
        predictions = np.load(model_dir / f"{model.lower()}_predictions.npy")
        labels = np.load(model_dir / f"{model.lower()}_labels.npy")
        
        with open(model_dir / "test_results.json", 'r') as f:
            test_results = json.load(f)
        
        history_path = model_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                histories[model] = json.load(f)
        
        results[model] = {
            "errors": errors,
            "predictions": predictions,
            "labels": labels,
            **test_results
        }
    
    return results, histories


def plot_summary_table(all_results, output_dir):
    """Create a summary table of all 100 epoch experiments."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    columns = ['Experiment', 'Model', 'Mean (m)', 'Median (m)', '<5m', '<10m', '<20m', 'Max (m)']
    rows = []
    
    for exp_name, exp_info in EXPERIMENTS.items():
        if exp_name not in all_results:
            continue
        results = all_results[exp_name]
        for model, res in results.items():
            rows.append([
                exp_info["label"],
                model,
                f"{res['mean_error']:.2f}",
                f"{res['median_error']:.2f}",
                f"{res['within_5m']:.1f}%",
                f"{res['within_10m']:.1f}%",
                f"{res['within_20m']:.1f}%",
                f"{res['max_error']:.1f}",
            ])
    
    # Sort by mean error
    rows.sort(key=lambda x: float(x[2]))
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2E4057')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best result (first row after header)
    for i in range(len(columns)):
        table[(1, i)].set_facecolor('#90EE90')
        table[(1, i)].set_text_props(fontweight='bold')
    
    # Color model cells
    for i, row in enumerate(rows):
        model = row[1]
        table[(i+1, 1)].set_facecolor(MODEL_COLORS.get(model, '#CCCCCC'))
        table[(i+1, 1)].set_text_props(color='white', fontweight='bold')
    
    plt.title("100 Epochs Experiments - Full Summary (Ranked by Mean Error)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "01_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_summary_table.png")


def plot_lr_comparison(all_results, output_dir):
    """Compare learning rates side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ["mean_error", "median_error", "within_10m"]
    titles = ["Mean Error (lower is better)", "Median Error (lower is better)", "Within 10m % (higher is better)"]
    
    exp_labels = [EXPERIMENTS[exp]["label"] for exp in EXPERIMENTS if exp in all_results]
    x = np.arange(len(MODELS))
    width = 0.35
    
    for ax, metric, title in zip(axes, metrics, titles):
        for i, (exp_name, exp_info) in enumerate(EXPERIMENTS.items()):
            if exp_name not in all_results:
                continue
            results = all_results[exp_name]
            values = [results[m][metric] if m in results else 0 for m in MODELS]
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, values, width, label=exp_info["label"], 
                         color=exp_info["color"], alpha=0.8)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Learning Rate Comparison: 0.001 vs 0.0001 (100 Epochs, Full Data)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "02_lr_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_lr_comparison.png")


def plot_error_distributions(all_results, output_dir):
    """Plot error histograms for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (exp_name, exp_info) in zip(axes, EXPERIMENTS.items()):
        if exp_name not in all_results:
            continue
        results = all_results[exp_name]
        
        for model, res in results.items():
            ax.hist(res["errors"], bins=40, alpha=0.5, color=MODEL_COLORS[model],
                   label=f'{model} (μ={res["mean_error"]:.1f}m)', edgecolor='white')
        
        ax.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=5, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel("Error (meters)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{exp_info['label']}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)
    
    plt.suptitle("Error Distribution - 100 Epochs Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_error_distribution.png")


def plot_cdf_comparison(all_results, output_dir):
    """Plot CDF curves for all models across experiments."""
    plt.figure(figsize=(14, 8))
    
    linestyles = {"100ep_lr001_fulldata": "-", "100ep_lr0001_fulldata": "--"}
    
    for exp_name, exp_info in EXPERIMENTS.items():
        if exp_name not in all_results:
            continue
        results = all_results[exp_name]
        ls = linestyles.get(exp_name, "-")
        
        for model, res in results.items():
            sorted_errors = np.sort(res["errors"])
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
            plt.plot(sorted_errors, cdf, color=MODEL_COLORS[model], linewidth=2.5,
                    linestyle=ls, label=f'{model} ({exp_info["label"]})')
    
    plt.axvline(x=5, color='green', linestyle=':', linewidth=2, alpha=0.7, label='5m threshold')
    plt.axvline(x=10, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='10m threshold')
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Cumulative % of Samples", fontsize=12)
    plt.title("Cumulative Error Distribution - 100 Epochs Experiments\n(Solid: LR=0.001, Dashed: LR=0.0001)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 40)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_dir / "04_cdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_cdf_comparison.png")


def plot_best_model_detail(all_results, output_dir):
    """Detailed analysis of the best model: EfficientNet lr=0.001."""
    best_exp = "100ep_lr001_fulldata"
    best_model = "EfficientNet"
    
    if best_exp not in all_results or best_model not in all_results[best_exp]:
        print("  ⚠ Best model not found, skipping detailed analysis")
        return
    
    res = all_results[best_exp][best_model]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Error histogram with thresholds
    ax = axes[0, 0]
    n, bins, patches = ax.hist(res["errors"], bins=50, color=MODEL_COLORS[best_model], 
                                edgecolor='white', alpha=0.8)
    ax.axvline(x=5, color='green', linestyle='--', linewidth=2, label=f'5m ({res["within_5m"]:.1f}%)')
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, label=f'10m ({res["within_10m"]:.1f}%)')
    ax.axvline(x=res["mean_error"], color='red', linestyle='-', linewidth=2, 
               label=f'Mean: {res["mean_error"]:.2f}m')
    ax.axvline(x=res["median_error"], color='purple', linestyle='-', linewidth=2,
               label=f'Median: {res["median_error"]:.2f}m')
    ax.set_xlabel("Error (meters)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    
    # 2. Geographic heatmap
    ax = axes[0, 1]
    scatter = ax.scatter(res["labels"][:, 1], res["labels"][:, 0], 
                        c=res["errors"], cmap='RdYlGn_r', s=20, alpha=0.8,
                        vmin=0, vmax=np.percentile(res["errors"], 95))
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Geographic Error Distribution", fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Error (m)')
    ax.grid(True, alpha=0.3)
    
    # 3. Prediction arrows (sample of worst predictions)
    ax = axes[1, 0]
    worst_idx = np.argsort(res["errors"])[-20:]  # 20 worst
    for idx in worst_idx:
        ax.annotate('', 
                   xy=(res["predictions"][idx, 1], res["predictions"][idx, 0]),
                   xytext=(res["labels"][idx, 1], res["labels"][idx, 0]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.6, lw=1.5))
    
    # Also show best predictions
    best_idx = np.argsort(res["errors"])[:20]
    for idx in best_idx:
        ax.scatter(res["labels"][idx, 1], res["labels"][idx, 0], 
                  c='green', s=30, alpha=0.6, marker='o')
    
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Prediction Arrows (Red: 20 worst, Green: 20 best)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Error percentiles
    ax = axes[1, 1]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = [np.percentile(res["errors"], p) for p in percentiles]
    bars = ax.bar([f'P{p}' for p in percentiles], pct_values, color=MODEL_COLORS[best_model], 
                  edgecolor='white', alpha=0.8)
    for bar, val in zip(bars, pct_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{val:.1f}m', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel("Error (meters)", fontsize=11)
    ax.set_title("Error Percentiles", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Best Model: {best_model} (100ep, LR=0.001, Full Data)\n"
                 f"Mean: {res['mean_error']:.2f}m | Median: {res['median_error']:.2f}m | "
                 f"Within 10m: {res['within_10m']:.1f}%", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "05_best_model_detail.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_best_model_detail.png")


def plot_training_curves(all_histories, output_dir):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (exp_name, exp_info) in zip(axes, EXPERIMENTS.items()):
        if exp_name not in all_histories:
            continue
        histories = all_histories[exp_name]
        
        for model, history in histories.items():
            if "val_loss" in history:
                epochs = range(1, len(history["val_loss"]) + 1)
                ax.plot(epochs, history["val_loss"], color=MODEL_COLORS[model], 
                       linewidth=2, label=model)
        
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Error (meters)", fontsize=11)
        ax.set_title(f"{exp_info['label']}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
    
    plt.suptitle("Training Curves - 100 Epochs Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "06_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_training_curves.png")


def plot_accuracy_thresholds(all_results, output_dir):
    """Plot accuracy at different thresholds."""
    thresholds = [3, 5, 10, 15, 20, 30]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (exp_name, exp_info) in zip(axes, EXPERIMENTS.items()):
        if exp_name not in all_results:
            continue
        results = all_results[exp_name]
        
        x = np.arange(len(thresholds))
        width = 0.25
        
        for i, model in enumerate(MODELS):
            if model not in results:
                continue
            errors = results[model]["errors"]
            accuracies = [(errors <= t).sum() / len(errors) * 100 for t in thresholds]
            offset = width * (i - 1)
            bars = ax.bar(x + offset, accuracies, width, label=model, 
                         color=MODEL_COLORS[model], alpha=0.8)
        
        ax.set_xlabel("Distance Threshold (meters)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"{exp_info['label']}", fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'≤{t}m' for t in thresholds], fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
    
    plt.suptitle("Accuracy at Different Distance Thresholds", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_accuracy_thresholds.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_accuracy_thresholds.png")


def generate_summary_log(all_results, output_dir):
    """Generate a text summary of results."""
    with open(output_dir / "analysis_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("100 EPOCHS EXPERIMENTS ANALYSIS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Collect all results
        all_models = []
        for exp_name, exp_info in EXPERIMENTS.items():
            if exp_name not in all_results:
                continue
            for model, res in all_results[exp_name].items():
                all_models.append({
                    "exp": exp_info["label"],
                    "model": model,
                    "mean": res["mean_error"],
                    "median": res["median_error"],
                    "within_5m": res["within_5m"],
                    "within_10m": res["within_10m"],
                    "within_20m": res["within_20m"],
                })
        
        # Sort by mean error
        all_models.sort(key=lambda x: x["mean"])
        
        f.write("RANKING BY MEAN ERROR:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6}{'Experiment':<20}{'Model':<15}{'Mean':<10}{'Median':<10}{'<10m':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, m in enumerate(all_models, 1):
            f.write(f"{i:<6}{m['exp']:<20}{m['model']:<15}{m['mean']:<10.2f}{m['median']:<10.2f}{m['within_10m']:<10.1f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("=" * 70 + "\n")
        
        best = all_models[0]
        f.write(f"\n🏆 BEST MODEL: {best['model']} ({best['exp']})\n")
        f.write(f"   Mean Error: {best['mean']:.2f}m\n")
        f.write(f"   Median Error: {best['median']:.2f}m\n")
        f.write(f"   Within 5m: {best['within_5m']:.1f}%\n")
        f.write(f"   Within 10m: {best['within_10m']:.1f}%\n")
        f.write(f"   Within 20m: {best['within_20m']:.1f}%\n")
        
        # Compare LRs
        f.write("\n\nLEARNING RATE COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        lr001_best = min([m for m in all_models if "0.001" in m["exp"]], key=lambda x: x["mean"])
        lr0001_best = min([m for m in all_models if "0.0001" in m["exp"]], key=lambda x: x["mean"])
        
        f.write(f"Best with LR=0.001:  {lr001_best['model']} - {lr001_best['mean']:.2f}m\n")
        f.write(f"Best with LR=0.0001: {lr0001_best['model']} - {lr0001_best['mean']:.2f}m\n")
        f.write(f"\n→ LR=0.001 is {lr0001_best['mean'] - lr001_best['mean']:.2f}m better for best model\n")
    
    print("  ✓ analysis_summary.txt")


def main():
    print("=" * 70)
    print("100 EPOCHS EXPERIMENTS ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all experiment results
    all_results = {}
    all_histories = {}
    
    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\nLoading {exp_name}...")
        results, histories = load_experiment_results(exp_info["path"])
        if results:
            all_results[exp_name] = results
            all_histories[exp_name] = histories
            print(f"  Loaded {len(results)} models")
    
    if not all_results:
        print("No results found!")
        return
    
    print("\nGenerating plots...")
    
    # Generate all plots
    plot_summary_table(all_results, OUTPUT_DIR)
    plot_lr_comparison(all_results, OUTPUT_DIR)
    plot_error_distributions(all_results, OUTPUT_DIR)
    plot_cdf_comparison(all_results, OUTPUT_DIR)
    plot_best_model_detail(all_results, OUTPUT_DIR)
    plot_training_curves(all_histories, OUTPUT_DIR)
    plot_accuracy_thresholds(all_results, OUTPUT_DIR)
    generate_summary_log(all_results, OUTPUT_DIR)
    
    print(f"\n✅ Analysis complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
