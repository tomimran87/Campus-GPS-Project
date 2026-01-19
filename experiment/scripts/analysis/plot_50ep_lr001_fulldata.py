#!/usr/bin/env python3
"""
Generate comparison plots for 50ep_lr001_fulldata experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


BASE_DIR = Path("testing/50ep_lr001_fulldata")
MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]
COLORS = {"ResNet18": "#e41a1c", "EfficientNet": "#377eb8", "ConvNeXt": "#4daf4a"}
EXPERIMENT_LABEL = "50 Epochs, LR=0.001, Full Data"


def load_results():
    results = {}
    histories = {}
    
    for model in MODELS:
        model_dir = BASE_DIR / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
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


def generate_all_plots(results, histories):
    output_dir = BASE_DIR / "comparison" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. Summary Table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    columns = ['Model', 'Mean (m)', 'Median (m)', '<5m', '<10m', '<20m', 'Epochs', 'Max (m)']
    rows = []
    for model, res in results.items():
        rows.append([
            model,
            f"{res['mean_error']:.2f}",
            f"{res['median_error']:.2f}",
            f"{res['within_5m']:.1f}%",
            f"{res['within_10m']:.1f}%",
            f"{res['within_20m']:.1f}%",
            str(res['final_epoch']),
            f"{res['max_error']:.1f}",
        ])
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, model in enumerate(results.keys()):
        table[(i+1, 0)].set_facecolor(COLORS[model])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f"Model Comparison Summary\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "00_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 00_summary_table.png")
    
    # 1. Learning curves
    if histories:
        n_models = len(histories)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for ax, (model, history) in zip(axes, histories.items()):
            epochs = range(1, len(history["val_loss"]) + 1)
            ax.plot(epochs, history["val_loss"], color=COLORS[model], linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Validation Error (m)", fontsize=11)
            ax.set_title(f"{model}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Validation Error During Training\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "01_learning_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 01_learning_curves.png")
    
    # 2. Error histogram
    plt.figure(figsize=(12, 7))
    for model, res in results.items():
        plt.hist(res["errors"], bins=40, alpha=0.5, color=COLORS[model],
                 label=f'{model} (μ={res["mean_error"]:.1f}m)', edgecolor='white')
    plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='10m')
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "02_error_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_error_histogram.png")
    
    # 3. CDF
    plt.figure(figsize=(12, 7))
    for model, res in results.items():
        sorted_errors = np.sort(res["errors"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cdf, color=COLORS[model], linewidth=2,
                 label=f'{model} (median={res["median_error"]:.1f}m)')
    plt.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10m')
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Cumulative %", fontsize=12)
    plt.title(f"Cumulative Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_cdf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_error_cdf.png")
    
    # 4. Box plot
    plt.figure(figsize=(10, 7))
    names = list(results.keys())
    error_data = [results[n]["errors"] for n in names]
    bp = plt.boxplot(error_data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)
    plt.ylabel("Error (meters)", fontsize=12)
    plt.title(f"Error Box Plot\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "04_error_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_error_boxplot.png")
    
    # 5. Metrics comparison
    metrics = ["mean_error", "median_error", "within_10m", "within_20m"]
    metric_labels = ["Mean Error (m)", "Median Error (m)", "Within 10m (%)", "Within 20m (%)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[m][metric] for m in results.keys()]
        bars = ax.bar(results.keys(), values, color=[COLORS[m] for m in results.keys()], edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Model Performance\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "05_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_metrics_comparison.png")
    
    # 6. Geographic heatmaps
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    all_errors = np.concatenate([res["errors"] for res in results.values()])
    vmax = np.percentile(all_errors, 95)
    
    for ax, (model, res) in zip(axes, results.items()):
        scatter = ax.scatter(
            res["labels"][:, 1], res["labels"][:, 0],
            c=res["errors"], cmap='RdYlGn_r', s=30, alpha=0.7, vmin=0, vmax=vmax
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{model}\n(Mean: {res['mean_error']:.2f}m)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Error (meters)', fontsize=11)
    
    plt.suptitle(f"Geographic Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_dir / "06_geographic_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_geographic_heatmaps.png")
    
    print(f"\n✅ All plots saved to {output_dir}")


def main():
    print("=" * 60)
    print(f"Generating plots for: {EXPERIMENT_LABEL}")
    print("=" * 60)
    
    results, histories = load_results()
    print(f"Loaded {len(results)} models")
    
    generate_all_plots(results, histories)


if __name__ == "__main__":
    main()
