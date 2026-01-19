#!/usr/bin/env python3
"""
Geographic Analysis Plots for 50ep_lr001_fulldata experiment.
Generates comprehensive spatial analysis plots.
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
    """Load all model results."""
    results = {}
    
    for model in MODELS:
        model_dir = BASE_DIR / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
            print(f"  ⚠ {model} results not found")
            continue
        
        errors = np.load(errors_path)
        predictions = np.load(model_dir / f"{model.lower()}_predictions.npy")
        labels = np.load(model_dir / f"{model.lower()}_labels.npy")
        
        with open(model_dir / "test_results.json", 'r') as f:
            test_results = json.load(f)
        
        results[model] = {
            "errors": errors,
            "predictions": predictions,
            "labels": labels,
            **test_results
        }
    
    return results


def plot_error_heatmaps(results, output_dir):
    """Create geographic heatmaps for each model."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model, res) in zip(axes, results.items()):
        labels = res["labels"]
        errors = res["errors"]
        
        scatter = ax.scatter(
            labels[:, 1], labels[:, 0],
            c=errors,
            cmap='RdYlGn_r',
            s=30,
            alpha=0.7
        )
        
        plt.colorbar(scatter, ax=ax, label='Error (m)')
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{model}\nMean: {res['mean_error']:.2f}m", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Geographic Error Distribution\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "01_error_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_error_heatmaps.png")


def plot_error_directions(results, output_dir):
    """Quiver plot showing prediction error directions."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model, res) in zip(axes, results.items()):
        labels = res["labels"]
        predictions = res["predictions"]
        errors = res["errors"]
        
        # Calculate direction vectors (pred - label)
        dlat = predictions[:, 0] - labels[:, 0]
        dlon = predictions[:, 1] - labels[:, 1]
        
        # Normalize for visualization
        magnitudes = np.sqrt(dlat**2 + dlon**2)
        magnitudes[magnitudes == 0] = 1
        dlat_norm = dlat / magnitudes * 0.0005  # Scale for visibility
        dlon_norm = dlon / magnitudes * 0.0005
        
        ax.scatter(labels[:, 1], labels[:, 0], c=errors, cmap='RdYlGn_r', s=20, alpha=0.5, zorder=1)
        ax.quiver(labels[:, 1], labels[:, 0], dlon_norm, dlat_norm, 
                 errors, cmap='RdYlGn_r', scale=0.05, width=0.003, alpha=0.8, zorder=2)
        
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{model} Error Directions", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Prediction Error Directions\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "02_error_directions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_error_directions.png")


def plot_error_vs_distance_from_center(results, output_dir):
    """Plot error vs distance from dataset center."""
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


def plot_model_disagreement_map(results, output_dir):
    """Map showing where models disagree most."""
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


def plot_prediction_scatter(results, output_dir):
    """Scatter plot: Predictions vs Actual for lat and lon."""
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    
    for col, (model, res) in enumerate(results.items()):
        labels = res["labels"]
        predictions = res["predictions"]
        
        # Latitude
        ax = axes[0, col]
        ax.scatter(labels[:, 0], predictions[:, 0], alpha=0.5, s=10, c=COLORS[model])
        lim = [min(labels[:, 0].min(), predictions[:, 0].min()),
               max(labels[:, 0].max(), predictions[:, 0].max())]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel("True Latitude", fontsize=10)
        ax.set_ylabel("Predicted Latitude", fontsize=10)
        ax.set_title(f"{model} - Latitude", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Longitude
        ax = axes[1, col]
        ax.scatter(labels[:, 1], predictions[:, 1], alpha=0.5, s=10, c=COLORS[model])
        lim = [min(labels[:, 1].min(), predictions[:, 1].min()),
               max(labels[:, 1].max(), predictions[:, 1].max())]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel("True Longitude", fontsize=10)
        ax.set_ylabel("Predicted Longitude", fontsize=10)
        ax.set_title(f"{model} - Longitude", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Prediction vs Actual\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_prediction_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_prediction_scatter.png")


def plot_error_by_region(results, output_dir):
    """Quadrant-based error analysis."""
    first_model = list(results.keys())[0]
    labels = results[first_model]["labels"]
    
    center_lat = np.median(labels[:, 0])
    center_lon = np.median(labels[:, 1])
    
    regions = {
        "NE": lambda l: (l[:, 0] >= center_lat) & (l[:, 1] >= center_lon),
        "NW": lambda l: (l[:, 0] >= center_lat) & (l[:, 1] < center_lon),
        "SE": lambda l: (l[:, 0] < center_lat) & (l[:, 1] >= center_lon),
        "SW": lambda l: (l[:, 0] < center_lat) & (l[:, 1] < center_lon),
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(regions))
    width = 0.25
    
    for i, (model, res) in enumerate(results.items()):
        region_errors = []
        for region_name, region_mask in regions.items():
            mask = region_mask(res["labels"])
            region_errors.append(np.mean(res["errors"][mask]))
        
        bars = ax.bar(x + i * width, region_errors, width, label=model, color=COLORS[model])
        for bar, val in zip(bars, region_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Campus Region", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title(f"Error by Campus Region\n({EXPERIMENT_LABEL})", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(regions.keys()), fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "08_error_by_region.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_error_by_region.png")


def main():
    print("=" * 60)
    print("Geographic Analysis: 50 Epochs, LR=0.001, Full Data")
    print("=" * 60)
    
    # Setup output directory
    output_dir = BASE_DIR / "geographic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n📂 Loading results...")
    results = load_results()
    print(f"  Loaded {len(results)} models: {list(results.keys())}")
    
    # Generate plots
    print("\n📊 Generating geographic plots...")
    plot_error_heatmaps(results, output_dir)
    plot_error_directions(results, output_dir)
    plot_error_vs_distance_from_center(results, output_dir)
    plot_grid_error_analysis(results, output_dir)
    plot_best_worst_locations(results, output_dir)
    plot_model_disagreement_map(results, output_dir)
    plot_prediction_scatter(results, output_dir)
    plot_error_by_region(results, output_dir)
    
    print(f"\n✅ All geographic plots saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
