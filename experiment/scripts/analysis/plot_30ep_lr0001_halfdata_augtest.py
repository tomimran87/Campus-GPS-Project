#!/usr/bin/env python3
"""
Comprehensive Plotting Script for 30ep_lr0001_halfdata_augtest Experiment
==========================================================================

Generates ALL plots as suggested:

1. TRAINING DYNAMICS:
   - Learning curves (train vs val loss)
   - Learning rate schedule
   
2. MODEL COMPARISON:
   - Error histogram overlay
   - Box plot comparison
   - CDF comparison
   - Error correlation matrix
   - Model disagreement map
   
3. GEOGRAPHIC ANALYSIS:
   - Error heatmaps per model
   - Direction of errors (arrows)
   - Error by distance from center
   - Grid-based error analysis
   - Best/worst location analysis
   
4. DATA ANALYSIS:
   - Sample distribution map
   - Hard sample analysis
   
5. STATISTICAL PLOTS:
   - Error percentiles chart
   - Summary table
   - Confidence intervals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
from scipy import stats
from datetime import datetime


# ============================================
# Configuration
# ============================================
BASE_DIR = Path("testing/30ep_lr0001_halfdata_augtest")

MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]

COLORS = {
    "ResNet18": "#e41a1c",
    "EfficientNet": "#377eb8",
    "ConvNeXt": "#4daf4a",
}

EXPERIMENT_LABEL = "30 Epochs, LR=0.0001, Half Data, Augmented Test"


def load_results():
    """Load all model results."""
    results = {}
    histories = {}
    
    for model in MODELS:
        model_dir = BASE_DIR / model
        
        # Load errors and predictions
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        preds_path = model_dir / f"{model.lower()}_predictions.npy"
        labels_path = model_dir / f"{model.lower()}_labels.npy"
        
        if not errors_path.exists():
            print(f"⚠ Missing data for {model}")
            continue
        
        errors = np.load(errors_path)
        predictions = np.load(preds_path)
        labels = np.load(labels_path)
        
        # Load test results
        with open(model_dir / "test_results.json", 'r') as f:
            test_results = json.load(f)
        
        # Load training history
        history_path = model_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            histories[model] = history
        
        results[model] = {
            "errors": errors,
            "predictions": predictions,
            "labels": labels,
            **test_results
        }
    
    return results, histories


# ============================================
# 1. TRAINING DYNAMICS PLOTS
# ============================================
def plot_learning_curves(histories, output_dir):
    """Plot training vs validation loss curves."""
    output_dir = output_dir / "training_dynamics"
    output_dir.mkdir(exist_ok=True)
    
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model, history) in zip(axes, histories.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        
        ax.plot(epochs, history["train_loss"], 'b-', label='Train Loss (L1)', linewidth=2)
        ax.plot(epochs, history["val_loss"], 'r-', label='Val Error (m)', linewidth=2)
        
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss / Error", fontsize=11)
        ax.set_title(f"{model} Learning Curve", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Learning Curves\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "01_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_learning_curves.png")


def plot_lr_schedule(histories, output_dir):
    """Plot learning rate schedule."""
    output_dir = output_dir / "training_dynamics"
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for model, history in histories.items():
        epochs = range(1, len(history["lr"]) + 1)
        plt.plot(epochs, history["lr"], color=COLORS[model], label=model, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title(f"Learning Rate Schedule\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / "02_lr_schedule.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_lr_schedule.png")


def plot_combined_learning(histories, output_dir):
    """Plot all models' validation loss on same graph."""
    output_dir = output_dir / "training_dynamics"
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for model, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        plt.plot(epochs, history["val_loss"], color=COLORS[model], label=model, linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Error (meters)", fontsize=12)
    plt.title(f"Validation Error During Training\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_combined_val_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_combined_val_loss.png")


# ============================================
# 2. MODEL COMPARISON PLOTS
# ============================================
def plot_error_histogram_overlay(results, output_dir):
    """Overlay histograms for all models."""
    output_dir = output_dir / "comparison"
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    for model, res in results.items():
        plt.hist(res["errors"], bins=40, alpha=0.5, color=COLORS[model], 
                 label=f'{model} (μ={res["mean_error"]:.1f}m)', edgecolor='white')
    
    plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='10m threshold')
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Error Distribution Comparison\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, None)
    plt.tight_layout()
    plt.savefig(output_dir / "01_error_histogram_overlay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_error_histogram_overlay.png")


def plot_boxplot_comparison(results, output_dir):
    """Box plot comparison of errors."""
    output_dir = output_dir / "comparison"
    
    plt.figure(figsize=(10, 7))
    
    names = list(results.keys())
    error_data = [results[n]["errors"] for n in names]
    
    bp = plt.boxplot(error_data, tick_labels=names, patch_artist=True, showfliers=True)
    
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)
    
    plt.ylabel("Error (meters)", fontsize=12)
    plt.title(f"Error Box Plot Comparison\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "02_error_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_error_boxplot.png")


def plot_violin_comparison(results, output_dir):
    """Violin plot comparison."""
    output_dir = output_dir / "comparison"
    
    plt.figure(figsize=(10, 7))
    
    names = list(results.keys())
    error_data = [results[n]["errors"] for n in names]
    
    parts = plt.violinplot(error_data, positions=range(len(names)), showmedians=True, showextrema=True)
    
    for i, (pc, name) in enumerate(zip(parts['bodies'], names)):
        pc.set_facecolor(COLORS[name])
        pc.set_alpha(0.7)
    
    plt.xticks(range(len(names)), names, fontsize=11)
    plt.ylabel("Error (meters)", fontsize=12)
    plt.title(f"Error Violin Plot\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_violin.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_error_violin.png")


def plot_cdf_comparison(results, output_dir):
    """CDF comparison plot."""
    output_dir = output_dir / "comparison"
    
    plt.figure(figsize=(12, 7))
    
    for model, res in results.items():
        sorted_errors = np.sort(res["errors"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cdf, color=COLORS[model], linewidth=2, 
                 label=f'{model} (median={res["median_error"]:.1f}m)')
    
    # Reference lines
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10m')
    plt.axvline(x=20, color='black', linestyle=':', alpha=0.5, label='20m')
    
    plt.xlabel("Error (meters)", fontsize=12)
    plt.ylabel("Cumulative Percentage (%)", fontsize=12)
    plt.title(f"Cumulative Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 80)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_dir / "04_error_cdf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_error_cdf.png")


def plot_error_correlation_matrix(results, output_dir):
    """Correlation matrix of errors between models."""
    output_dir = output_dir / "comparison"
    
    names = list(results.keys())
    n = len(names)
    
    # Compute correlation matrix
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            corr = np.corrcoef(results[m1]["errors"], results[m2]["errors"])[0, 1]
            corr_matrix[i, j] = corr
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    
    # Add correlation values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.title(f"Error Correlation Between Models\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "05_error_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_error_correlation.png")


def plot_model_disagreement_map(results, output_dir):
    """Map showing where models disagree most."""
    output_dir = output_dir / "comparison"
    
    models = list(results.keys())
    if len(models) < 2:
        return
    
    # Use first model's labels as reference
    labels = results[models[0]]["labels"]
    
    # Compute disagreement (std of errors across models)
    all_errors = np.array([results[m]["errors"] for m in models])
    disagreement = np.std(all_errors, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        labels[:, 1], labels[:, 0],
        c=disagreement,
        cmap='YlOrRd',
        s=50,
        alpha=0.7
    )
    
    plt.colorbar(scatter, ax=ax, label='Model Disagreement (Std of Errors)')
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Model Disagreement Map\n(High = Models Predict Differently)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "06_model_disagreement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_model_disagreement.png")


def plot_metrics_bar_chart(results, output_dir):
    """Bar chart of key metrics."""
    output_dir = output_dir / "comparison"
    
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
    
    plt.suptitle(f"Model Performance Metrics\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_metrics_comparison.png")


def plot_accuracy_thresholds(results, output_dir):
    """Grouped bar chart of accuracy at different thresholds."""
    output_dir = output_dir / "comparison"
    
    thresholds = [5, 10, 15, 20, 30, 50]
    threshold_keys = [f"within_{t}m" for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, (model, res) in enumerate(results.items()):
        values = [res.get(k, np.mean(res["errors"] < t) * 100) 
                  for k, t in zip(threshold_keys, thresholds)]
        bars = ax.bar(x + i * width, values, width, label=model, color=COLORS[model], edgecolor='white')
    
    ax.set_xlabel("Error Threshold (meters)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy at Different Thresholds\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"<{t}m" for t in thresholds], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "08_accuracy_thresholds.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_accuracy_thresholds.png")


# ============================================
# 3. GEOGRAPHIC ANALYSIS PLOTS
# ============================================
def plot_geographic_error_heatmaps(results, output_dir):
    """Side-by-side geographic error heatmaps."""
    output_dir = output_dir / "geographic"
    output_dir.mkdir(exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    # Find global error range for consistent colorbar
    all_errors = np.concatenate([res["errors"] for res in results.values()])
    vmax = np.percentile(all_errors, 95)
    
    for ax, (model, res) in zip(axes, results.items()):
        scatter = ax.scatter(
            res["labels"][:, 1], res["labels"][:, 0],
            c=res["errors"],
            cmap='RdYlGn_r',
            s=30,
            alpha=0.7,
            vmin=0, vmax=vmax
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{model}\n(Mean: {res['mean_error']:.2f}m)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # External colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Error (meters)', fontsize=11)
    
    plt.suptitle(f"Geographic Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_dir / "01_geographic_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_geographic_heatmaps.png")


def plot_error_direction_arrows(results, output_dir):
    """Plot arrows showing prediction direction errors."""
    output_dir = output_dir / "geographic"
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model, res) in zip(axes, results.items()):
        labels = res["labels"]
        preds = res["predictions"]
        errors = res["errors"]
        
        # Plot arrows for a subset (to avoid clutter)
        n_arrows = min(100, len(labels))
        indices = np.random.choice(len(labels), n_arrows, replace=False)
        
        # Color by error magnitude
        colors = plt.cm.RdYlGn_r(errors[indices] / errors.max())
        
        for i, idx in enumerate(indices):
            ax.annotate('', 
                xy=(preds[idx, 1], preds[idx, 0]),
                xytext=(labels[idx, 1], labels[idx, 0]),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7)
            )
        
        # Plot actual locations
        ax.scatter(labels[:, 1], labels[:, 0], c='blue', s=10, alpha=0.3, label='Actual')
        
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{model} - Prediction Direction\n(Arrow: Actual → Predicted)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Prediction Error Directions\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "02_error_directions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_error_directions.png")


def plot_error_vs_distance_from_center(results, output_dir):
    """Plot error vs distance from dataset center."""
    output_dir = output_dir / "geographic"
    
    # Calculate center
    first_model = list(results.keys())[0]
    labels = results[first_model]["labels"]
    center_lat = np.mean(labels[:, 0])
    center_lon = np.mean(labels[:, 1])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model, res in results.items():
        # Calculate distance from center for each point
        distances_from_center = np.sqrt(
            (res["labels"][:, 0] - center_lat)**2 + 
            (res["labels"][:, 1] - center_lon)**2
        ) * 111000  # Approximate conversion to meters
        
        # Bin the data
        bins = np.linspace(0, distances_from_center.max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        binned_errors = []
        for i in range(len(bins) - 1):
            mask = (distances_from_center >= bins[i]) & (distances_from_center < bins[i+1])
            if np.sum(mask) > 0:
                binned_errors.append(np.mean(res["errors"][mask]))
            else:
                binned_errors.append(np.nan)
        
        ax.plot(bin_centers, binned_errors, color=COLORS[model], linewidth=2, 
                marker='o', markersize=8, label=model)
    
    ax.set_xlabel("Distance from Campus Center (meters)", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title(f"Error vs Distance from Center\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_vs_distance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_error_vs_distance.png")


def plot_grid_error_analysis(results, output_dir):
    """Grid-based error analysis."""
    output_dir = output_dir / "geographic"
    
    GRID_SIZE = 6
    first_model = list(results.keys())[0]
    labels = results[first_model]["labels"]
    
    lat_bins = np.linspace(labels[:, 0].min(), labels[:, 0].max(), GRID_SIZE + 1)
    lon_bins = np.linspace(labels[:, 1].min(), labels[:, 1].max(), GRID_SIZE + 1)
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model, res) in zip(axes, results.items()):
        grid_errors = np.zeros((GRID_SIZE, GRID_SIZE))
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                mask = (
                    (res["labels"][:, 0] >= lat_bins[i]) & 
                    (res["labels"][:, 0] < lat_bins[i+1]) &
                    (res["labels"][:, 1] >= lon_bins[j]) & 
                    (res["labels"][:, 1] < lon_bins[j+1])
                )
                if np.sum(mask) > 0:
                    grid_errors[i, j] = np.mean(res["errors"][mask])
                else:
                    grid_errors[i, j] = np.nan
        
        im = ax.imshow(grid_errors, cmap='RdYlGn_r', origin='lower', aspect='auto')
        plt.colorbar(im, ax=ax, label='Mean Error (m)')
        
        ax.set_xlabel("Longitude Grid", fontsize=11)
        ax.set_ylabel("Latitude Grid", fontsize=11)
        ax.set_title(f"{model} Grid Error", fontsize=12, fontweight='bold')
    
    plt.suptitle(f"Grid-Based Error Analysis\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "04_grid_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_grid_error.png")


def plot_best_worst_locations(results, output_dir):
    """Plot best and worst prediction locations."""
    output_dir = output_dir / "geographic"
    
    TOP_N = 15
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    for model, res in results.items():
        errors = res["errors"]
        labels = res["labels"]
        
        # Best predictions
        best_idx = np.argsort(errors)[:TOP_N]
        ax.scatter(labels[best_idx, 1], labels[best_idx, 0],
                  color=COLORS[model], marker='o', s=100, alpha=0.7,
                  label=f'{model} Best', edgecolors='green', linewidths=2)
        
        # Worst predictions
        worst_idx = np.argsort(errors)[-TOP_N:]
        ax.scatter(labels[worst_idx, 1], labels[worst_idx, 0],
                  color=COLORS[model], marker='x', s=100, alpha=0.7,
                  label=f'{model} Worst', linewidths=2)
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Best (○) and Worst (✕) Prediction Locations\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "05_best_worst_locations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_best_worst_locations.png")


# ============================================
# 4. STATISTICAL PLOTS
# ============================================
def plot_error_percentiles(results, output_dir):
    """Plot error percentiles for all models."""
    output_dir = output_dir / "comparison"
    
    percentiles = [50, 75, 90, 95, 99]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(percentiles))
    width = 0.25
    
    for i, (model, res) in enumerate(results.items()):
        values = [np.percentile(res["errors"], p) for p in percentiles]
        bars = ax.bar(x + i * width, values, width, label=model, color=COLORS[model], edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Percentile", fontsize=12)
    ax.set_ylabel("Error (meters)", fontsize=12)
    ax.set_title(f"Error Percentiles\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'P{p}' for p in percentiles], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "09_error_percentiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_error_percentiles.png")


def plot_summary_table(results, output_dir):
    """Create summary table as image."""
    output_dir = output_dir / "comparison"
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    columns = ['Model', 'Mean', 'Median', 'Std', 'P95', '<5m', '<10m', '<20m', 'Max']
    rows = []
    
    for model, res in results.items():
        rows.append([
            model,
            f"{res['mean_error']:.2f}m",
            f"{res['median_error']:.2f}m",
            f"{res['std_error']:.2f}m",
            f"{res.get('p95_error', np.percentile(res['errors'], 95)):.2f}m",
            f"{res['within_5m']:.1f}%",
            f"{res['within_10m']:.1f}%",
            f"{res['within_20m']:.1f}%",
            f"{res['max_error']:.1f}m",
        ])
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color model names
    for i, model in enumerate(results.keys()):
        table[(i+1, 0)].set_facecolor(COLORS[model])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f"Model Comparison Summary\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "00_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 00_summary_table.png")


def plot_confidence_intervals(results, output_dir):
    """Bootstrap confidence intervals for mean error."""
    output_dir = output_dir / "comparison"
    
    n_bootstrap = 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ci_data = []
    
    for model, res in results.items():
        errors = res["errors"]
        
        # Bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(errors, size=len(errors), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        mean = res["mean_error"]
        
        ci_data.append((model, mean, ci_lower, ci_upper))
    
    # Plot
    for i, (model, mean, ci_lower, ci_upper) in enumerate(ci_data):
        ax.errorbar(i, mean, yerr=[[mean - ci_lower], [ci_upper - mean]], 
                   fmt='o', color=COLORS[model], markersize=12, capsize=10, 
                   capthick=2, linewidth=2, label=f'{model}: {mean:.2f}m [{ci_lower:.2f}, {ci_upper:.2f}]')
    
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(list(results.keys()), fontsize=11)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title(f"95% Confidence Intervals for Mean Error\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "10_confidence_intervals.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 10_confidence_intervals.png")


# ============================================
# 5. HARD SAMPLE ANALYSIS
# ============================================
def analyze_hard_samples(results, output_dir):
    """Analyze samples that are hard for all models."""
    output_dir = output_dir / "comparison"
    
    models = list(results.keys())
    
    # Find samples with high error for all models
    min_len = min(len(results[m]["errors"]) for m in models)
    
    avg_errors = np.zeros(min_len)
    for model in models:
        avg_errors += results[model]["errors"][:min_len]
    avg_errors /= len(models)
    
    # Get indices of hardest samples
    hard_idx = np.argsort(avg_errors)[-20:]
    
    # Get labels for hardest samples
    labels = results[models[0]]["labels"][:min_len]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all samples lightly
    ax.scatter(labels[:, 1], labels[:, 0], c='lightgray', s=20, alpha=0.3, label='All samples')
    
    # Highlight hard samples
    scatter = ax.scatter(labels[hard_idx, 1], labels[hard_idx, 0], 
                         c=avg_errors[hard_idx], cmap='Reds', s=150, 
                         edgecolors='black', linewidths=2, label='Hardest samples')
    
    plt.colorbar(scatter, ax=ax, label='Average Error (m)')
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Hardest Samples (High Error Across All Models)\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "11_hard_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 11_hard_samples.png")


def plot_sample_distribution(results, output_dir):
    """Plot distribution of samples on map."""
    output_dir = output_dir / "geographic"
    
    first_model = list(results.keys())[0]
    labels = results[first_model]["labels"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 2D histogram
    h = ax.hist2d(labels[:, 1], labels[:, 0], bins=20, cmap='Blues')
    plt.colorbar(h[3], ax=ax, label='Sample Count')
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Test Sample Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "06_sample_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_sample_distribution.png")


# ============================================
# MAIN
# ============================================
def main():
    """Generate all plots."""
    print("=" * 70)
    print("COMPREHENSIVE PLOT GENERATION")
    print(f"Experiment: {EXPERIMENT_LABEL}")
    print("=" * 70)
    
    # Load results
    print("\nLoading results...")
    results, histories = load_results()
    
    if not results:
        print("❌ No results found. Run training first!")
        return
    
    print(f"Loaded {len(results)} models: {list(results.keys())}")
    
    # Create output directories
    (BASE_DIR / "training_dynamics").mkdir(exist_ok=True)
    (BASE_DIR / "comparison").mkdir(exist_ok=True)
    (BASE_DIR / "geographic").mkdir(exist_ok=True)
    
    # 1. Training Dynamics
    print("\n📊 Generating Training Dynamics Plots...")
    if histories:
        plot_learning_curves(histories, BASE_DIR)
        plot_lr_schedule(histories, BASE_DIR)
        plot_combined_learning(histories, BASE_DIR)
    
    # 2. Model Comparison
    print("\n📊 Generating Model Comparison Plots...")
    plot_summary_table(results, BASE_DIR)
    plot_error_histogram_overlay(results, BASE_DIR)
    plot_boxplot_comparison(results, BASE_DIR)
    plot_violin_comparison(results, BASE_DIR)
    plot_cdf_comparison(results, BASE_DIR)
    plot_error_correlation_matrix(results, BASE_DIR)
    plot_model_disagreement_map(results, BASE_DIR)
    plot_metrics_bar_chart(results, BASE_DIR)
    plot_accuracy_thresholds(results, BASE_DIR)
    plot_error_percentiles(results, BASE_DIR)
    plot_confidence_intervals(results, BASE_DIR)
    analyze_hard_samples(results, BASE_DIR)
    
    # 3. Geographic Analysis
    print("\n📊 Generating Geographic Analysis Plots...")
    plot_geographic_error_heatmaps(results, BASE_DIR)
    plot_error_direction_arrows(results, BASE_DIR)
    plot_error_vs_distance_from_center(results, BASE_DIR)
    plot_grid_error_analysis(results, BASE_DIR)
    plot_best_worst_locations(results, BASE_DIR)
    plot_sample_distribution(results, BASE_DIR)
    
    print("\n" + "=" * 70)
    print("✅ ALL PLOTS GENERATED!")
    print("=" * 70)
    print(f"\nOutput directories:")
    print(f"  • {BASE_DIR}/training_dynamics/ - Learning curves, LR schedule")
    print(f"  • {BASE_DIR}/comparison/ - Model comparison plots")
    print(f"  • {BASE_DIR}/geographic/ - Geographic analysis plots")
    
    # Count plots
    n_plots = 0
    for subdir in ["training_dynamics", "comparison", "geographic"]:
        path = BASE_DIR / subdir
        if path.exists():
            n_plots += len(list(path.glob("*.png")))
    
    print(f"\nTotal plots generated: {n_plots}")


if __name__ == "__main__":
    main()
