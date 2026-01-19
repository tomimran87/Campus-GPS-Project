"""
Generate Comparison Reports for 30epochs_lr0.0001_halfdata Experiment
======================================================================
Creates all individual and comparison plots for the half-data experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


# ============================================
# Configuration
# ============================================
BASE_DIR = Path("testing/30epochs_lr0.0001_halfdata")

MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]

COLORS = {
    "ResNet18": "#e41a1c",
    "EfficientNet": "#377eb8",
    "ConvNeXt": "#4daf4a",
}

EXPERIMENT_LABEL = "30 Epochs, LR=0.0001, 50% Data"


def load_results():
    """Load results from all models."""
    results = {}
    
    for model in MODELS:
        model_dir = BASE_DIR / model
        
        try:
            errors = np.load(model_dir / f"{model.lower()}_errors.npy")
            predictions = np.load(model_dir / f"{model.lower()}_predictions.npy")
            labels = np.load(model_dir / f"{model.lower()}_labels.npy")
            
            results[model] = {
                "errors": errors,
                "predictions": predictions,
                "labels": labels,
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "std_error": float(np.std(errors)),
                "min_error": float(np.min(errors)),
                "max_error": float(np.max(errors)),
                "within_5m": float(np.mean(errors < 5) * 100),
                "within_10m": float(np.mean(errors < 10) * 100),
                "within_20m": float(np.mean(errors < 20) * 100),
                "within_50m": float(np.mean(errors < 50) * 100),
                "test_samples": len(errors),
            }
            print(f"✓ Loaded {model}: {len(errors)} samples, mean={np.mean(errors):.2f}m")
        except FileNotFoundError as e:
            print(f"✗ {model}: Files not found")
    
    return results


# ============================================
# Individual Model Plots
# ============================================
def generate_individual_plots(results):
    """Generate individual model plots."""
    print("\nGenerating individual model plots...")
    
    for model, res in results.items():
        output_dir = BASE_DIR / model / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Error Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(res["errors"], bins=30, color=COLORS[model], edgecolor='white', alpha=0.7)
        plt.axvline(res["mean_error"], color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {res["mean_error"]:.2f}m')
        plt.axvline(res["median_error"], color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {res["median_error"]:.2f}m')
        plt.title(f"{model} - Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
        plt.xlabel("Error (meters)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "01_error_histogram.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. CDF
        plt.figure(figsize=(10, 6))
        sorted_errors = np.sort(res["errors"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cdf, color=COLORS[model], linewidth=2)
        plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10m threshold')
        plt.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='20m threshold')
        plt.title(f"{model} - Cumulative Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
        plt.xlabel("Error (meters)", fontsize=12)
        plt.ylabel("Cumulative Percentage (%)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(100, res["max_error"] * 1.1))
        plt.tight_layout()
        plt.savefig(output_dir / "02_error_cdf.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Scatter
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].scatter(res["labels"][:, 0], res["predictions"][:, 0], 
                       alpha=0.5, c=COLORS[model], s=20)
        min_lat = min(res["labels"][:, 0].min(), res["predictions"][:, 0].min())
        max_lat = max(res["labels"][:, 0].max(), res["predictions"][:, 0].max())
        axes[0].plot([min_lat, max_lat], [min_lat, max_lat], 'k--', linewidth=2, label='Perfect')
        axes[0].set_xlabel("True Latitude", fontsize=12)
        axes[0].set_ylabel("Predicted Latitude", fontsize=12)
        axes[0].set_title(f"{model} - Latitude", fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(res["labels"][:, 1], res["predictions"][:, 1], 
                       alpha=0.5, c=COLORS[model], s=20)
        min_lon = min(res["labels"][:, 1].min(), res["predictions"][:, 1].min())
        max_lon = max(res["labels"][:, 1].max(), res["predictions"][:, 1].max())
        axes[1].plot([min_lon, max_lon], [min_lon, max_lon], 'k--', linewidth=2, label='Perfect')
        axes[1].set_xlabel("True Longitude", fontsize=12)
        axes[1].set_ylabel("Predicted Longitude", fontsize=12)
        axes[1].set_title(f"{model} - Longitude", fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Prediction Scatter ({EXPERIMENT_LABEL})", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "03_prediction_scatter.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Geographic Error
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(res["labels"][:, 1], res["labels"][:, 0], 
                            c=res["errors"], cmap='RdYlGn_r', 
                            s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Error (meters)', fontsize=12)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title(f"{model} - Geographic Error\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "04_geographic_error.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Accuracy Thresholds
        thresholds = [5, 10, 15, 20, 30, 50]
        accuracies = [np.mean(res["errors"] < t) * 100 for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(thresholds)), accuracies, color=COLORS[model], edgecolor='white', alpha=0.8)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.xticks(range(len(thresholds)), [f'{t}m' for t in thresholds], fontsize=11)
        plt.xlabel("Error Threshold", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title(f"{model} - Accuracy at Thresholds\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / "05_accuracy_thresholds.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {model}: 5 plots generated")


# ============================================
# Comparison Plots (FIXED COLORBAR)
# ============================================
def generate_comparison_plots(results):
    """Generate comparison plots with fixed colorbar."""
    output_dir = BASE_DIR / "comparison" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating comparison plots...")
    
    # 0. Summary Table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    columns = ['Model', 'Mean (m)', 'Median (m)', 'Std (m)', '<5m', '<10m', '<20m', 'Max (m)']
    rows = []
    for model, res in results.items():
        rows.append([
            model,
            f"{res['mean_error']:.2f}",
            f"{res['median_error']:.2f}",
            f"{res['std_error']:.2f}",
            f"{res['within_5m']:.1f}%",
            f"{res['within_10m']:.1f}%",
            f"{res['within_20m']:.1f}%",
            f"{res['max_error']:.2f}",
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
    
    # 1. Histogram Overlay
    plt.figure(figsize=(12, 7))
    for model, res in results.items():
        plt.hist(res["errors"], bins=30, alpha=0.5, 
                label=f'{model} (mean={res["mean_error"]:.1f}m)', 
                color=COLORS[model])
    plt.title(f"Error Distribution - All Models\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_error_histogram_overlay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_error_histogram_overlay.png")
    
    # 2. Box Plot
    plt.figure(figsize=(10, 7))
    names = list(results.keys())
    error_data = [results[n]["errors"] for n in names]
    bp = plt.boxplot(error_data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS.get(name, "gray"))
        patch.set_alpha(0.6)
    plt.title(f"Error Box Plot\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.ylabel("Error (meters)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "02_error_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_error_boxplot.png")
    
    # 3. CDF Comparison
    plt.figure(figsize=(12, 7))
    for model, res in results.items():
        sorted_errors = np.sort(res["errors"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cdf, label=model, linewidth=2.5, color=COLORS[model])
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50%')
    plt.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90%')
    plt.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10m')
    plt.title(f"Cumulative Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Cumulative Percentage (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 80)
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_cdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_error_cdf_comparison.png")
    
    # 4. Metrics Bar Chart
    metrics = ["mean_error", "median_error", "within_10m", "within_20m"]
    metric_labels = ["Mean Error (m)", "Median Error (m)", "Within 10m (%)", "Within 20m (%)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        names = list(results.keys())
        values = [results[n][metric] for n in names]
        bars = ax.bar(names, values, color=[COLORS[n] for n in names], edgecolor='white', alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Model Performance Comparison\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "04_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_metrics_comparison.png")
    
    # 5. Grouped Accuracy Bar Chart
    thresholds = [5, 10, 15, 20, 30, 50]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(thresholds))
    width = 0.25
    
    for i, (model, res) in enumerate(results.items()):
        accuracies = [np.mean(res["errors"] < t) * 100 for t in thresholds]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x + offset, accuracies, width, label=model, color=COLORS[model], alpha=0.8)
    
    ax.set_xlabel("Error Threshold", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy at Different Thresholds\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}m' for t in thresholds], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "05_accuracy_grouped.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_accuracy_grouped.png")
    
    # 6. Geographic Comparison (FIXED: Colorbar outside plots)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models + 2, 7))
    
    if n_models == 1:
        axes = [axes]
    
    # Find global error range for consistent colorbar
    all_errors = np.concatenate([res["errors"] for res in results.values()])
    vmax = min(50, np.percentile(all_errors, 95))
    
    scatter = None
    for ax, (model, res) in zip(axes, results.items()):
        scatter = ax.scatter(res["labels"][:, 1], res["labels"][:, 0], 
                            c=res["errors"], cmap='RdYlGn_r', 
                            s=30, alpha=0.7, edgecolors='white', linewidth=0.3,
                            vmin=0, vmax=vmax)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{model}\n(Mean: {res['mean_error']:.1f}m)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Add colorbar on the right side, outside all plots
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Error (meters)', fontsize=11)
    
    plt.suptitle(f"Geographic Error Comparison\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.savefig(output_dir / "06_geographic_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_geographic_comparison.png")


def save_summary(results):
    """Save comparison summary."""
    comparison_dir = BASE_DIR / "comparison"
    
    with open(comparison_dir / "comparison_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"MODEL COMPARISON SUMMARY\n")
        f.write(f"{EXPERIMENT_LABEL}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for model, res in results.items():
            f.write(f"{model}:\n")
            f.write(f"  Mean Error:   {res['mean_error']:.2f}m\n")
            f.write(f"  Median Error: {res['median_error']:.2f}m\n")
            f.write(f"  Std Error:    {res['std_error']:.2f}m\n")
            f.write(f"  Within 5m:    {res['within_5m']:.1f}%\n")
            f.write(f"  Within 10m:   {res['within_10m']:.1f}%\n")
            f.write(f"  Within 20m:   {res['within_20m']:.1f}%\n")
            f.write(f"  Max Error:    {res['max_error']:.2f}m\n")
            f.write("\n")
        
        best_model = min(results.keys(), key=lambda x: results[x]["mean_error"])
        f.write("-" * 70 + "\n")
        f.write(f"BEST MODEL: {best_model} (Mean Error: {results[best_model]['mean_error']:.2f}m)\n")
        f.write("-" * 70 + "\n")
    
    print(f"\n✓ Saved comparison_summary.txt")


def main():
    print("=" * 70)
    print(f"GENERATING COMPARISON REPORTS")
    print(f"{EXPERIMENT_LABEL}")
    print("=" * 70)
    
    results = load_results()
    
    if len(results) == 0:
        print("\n✗ No results found. Run training first.")
        return
    
    generate_individual_plots(results)
    generate_comparison_plots(results)
    save_summary(results)
    
    print("\n" + "=" * 70)
    print("✓ ALL REPORTS GENERATED!")
    print(f"  Output: {BASE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
