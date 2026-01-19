#!/usr/bin/env python3
"""
Geographic Analysis for All Experiments
========================================

Generates geographic error analysis and scatter plots for all experiments
that have predictions and labels saved.

For each experiment:
1. Error scatter plot on map (predictions vs true locations)
2. Error heatmap by grid
3. Best/worst prediction locations
4. Geographic statistics

Output: Each experiment folder gets a 'geographic_analysis/' subfolder
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

ALL_EXPERIMENTS = {
    "30ep_Full": Path("testing/30epochs"),
    "30ep_Half": Path("testing/30epochs_lr0.0001_halfdata"),
    "30ep_Half_Aug": Path("testing/30ep_lr0001_halfdata_augtest"),
    "50ep_Half": Path("testing/50ep_lr0001_halfdata"),
    "50ep_Half_Aug": Path("testing/50ep_lr0001_halfdata_augtest"),
    "50ep_Full_lr001": Path("testing/50ep_lr001_fulldata"),
    "100ep_Half": Path("testing/100ep_lr0001_halfdata"),
    "100ep_Half_Aug": Path("testing/100ep_lr0001_halfdata_augtest"),
    "100ep_Full_lr001": Path("testing/100ep_lr001_fulldata"),
    "100ep_Full_lr0001": Path("testing/100ep_lr0001_fulldata"),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def load_experiment_data(exp_path, model):
    """Load predictions, labels, and errors for a model."""
    model_dir = exp_path / model
    
    errors_path = model_dir / f"{model.lower()}_errors.npy"
    preds_path = model_dir / f"{model.lower()}_predictions.npy"
    labels_path = model_dir / f"{model.lower()}_labels.npy"
    
    if not all(p.exists() for p in [errors_path, preds_path, labels_path]):
        return None
    
    return {
        "errors": np.load(errors_path),
        "predictions": np.load(preds_path),
        "labels": np.load(labels_path),
    }


def create_error_scatter(data, model, output_dir):
    """Create scatter plot of errors by location."""
    labels = data["labels"]
    errors = data["errors"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot with error as color
    scatter = ax.scatter(
        labels[:, 1], labels[:, 0],  # lon, lat
        c=errors, cmap='RdYlGn_r',
        s=50, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Error (m)')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{model} - Prediction Errors by Location')
    ax.grid(alpha=0.3)
    
    # Add stats
    stats_text = f"Mean: {np.mean(errors):.2f}m\nMedian: {np.median(errors):.2f}m\n<10m: {np.mean(errors<10)*100:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model.lower()}_error_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_prediction_arrows(data, model, output_dir):
    """Create plot showing prediction vectors (arrows from true to predicted)."""
    labels = data["labels"]
    predictions = data["predictions"]
    errors = data["errors"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sort by error to draw worst on top
    sort_idx = np.argsort(errors)
    
    # Plot arrows from true to predicted location
    for idx in sort_idx:
        error = errors[idx]
        true_lat, true_lon = labels[idx]
        pred_lat, pred_lon = predictions[idx]
        
        # Color by error
        if error < 5:
            color = 'green'
            alpha = 0.5
        elif error < 10:
            color = 'blue'
            alpha = 0.6
        elif error < 20:
            color = 'orange'
            alpha = 0.7
        else:
            color = 'red'
            alpha = 0.8
        
        ax.annotate('', xy=(pred_lon, pred_lat), xytext=(true_lon, true_lat),
                   arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1))
    
    # Plot true locations
    ax.scatter(labels[:, 1], labels[:, 0], c='black', s=20, marker='o', label='True', zorder=5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{model} - Prediction Vectors\n(Green <5m, Blue <10m, Orange <20m, Red >20m)')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model.lower()}_prediction_arrows.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_error_heatmap(data, model, output_dir, grid_size=10):
    """Create heatmap of errors by grid cell."""
    labels = data["labels"]
    errors = data["errors"]
    
    # Create grid
    lat_min, lat_max = labels[:, 0].min(), labels[:, 0].max()
    lon_min, lon_max = labels[:, 1].min(), labels[:, 1].max()
    
    lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)
    
    # Calculate mean error per grid cell
    error_grid = np.zeros((grid_size, grid_size))
    count_grid = np.zeros((grid_size, grid_size))
    
    for i in range(len(labels)):
        lat_idx = min(np.searchsorted(lat_bins, labels[i, 0]) - 1, grid_size - 1)
        lon_idx = min(np.searchsorted(lon_bins, labels[i, 1]) - 1, grid_size - 1)
        lat_idx = max(0, lat_idx)
        lon_idx = max(0, lon_idx)
        
        error_grid[lat_idx, lon_idx] += errors[i]
        count_grid[lat_idx, lon_idx] += 1
    
    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_error_grid = np.where(count_grid > 0, error_grid / count_grid, np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(mean_error_grid, cmap='RdYlGn_r', origin='lower',
                   extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto')
    
    plt.colorbar(im, label='Mean Error (m)')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{model} - Mean Error Heatmap by Area')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model.lower()}_error_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_best_worst_analysis(data, model, output_dir, n=20):
    """Analyze and plot best/worst predictions."""
    labels = data["labels"]
    predictions = data["predictions"]
    errors = data["errors"]
    
    # Sort by error
    sort_idx = np.argsort(errors)
    best_idx = sort_idx[:n]
    worst_idx = sort_idx[-n:][::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Best predictions
    ax = axes[0]
    ax.scatter(labels[best_idx, 1], labels[best_idx, 0], c='green', s=100, 
              marker='o', label='True', edgecolors='black')
    ax.scatter(predictions[best_idx, 1], predictions[best_idx, 0], c='blue', s=100, 
              marker='x', label='Predicted')
    
    for i, idx in enumerate(best_idx):
        ax.annotate(f'{errors[idx]:.1f}m', (labels[idx, 1], labels[idx, 0]), 
                   fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Top {n} Best Predictions\n(Mean: {np.mean(errors[best_idx]):.2f}m)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Worst predictions
    ax = axes[1]
    ax.scatter(labels[worst_idx, 1], labels[worst_idx, 0], c='red', s=100, 
              marker='o', label='True', edgecolors='black')
    ax.scatter(predictions[worst_idx, 1], predictions[worst_idx, 0], c='blue', s=100, 
              marker='x', label='Predicted')
    
    for i, idx in enumerate(worst_idx[:10]):  # Only label top 10 worst
        ax.annotate(f'{errors[idx]:.1f}m', (labels[idx, 1], labels[idx, 0]), 
                   fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Top {n} Worst Predictions\n(Mean: {np.mean(errors[worst_idx]):.2f}m)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'{model} - Best vs Worst Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model.lower()}_best_worst.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return stats for JSON
    return {
        "best_mean": float(np.mean(errors[best_idx])),
        "best_median": float(np.median(errors[best_idx])),
        "best_max": float(np.max(errors[best_idx])),
        "worst_mean": float(np.mean(errors[worst_idx])),
        "worst_median": float(np.median(errors[worst_idx])),
        "worst_min": float(np.min(errors[worst_idx])),
    }


def create_error_histogram_by_area(data, model, output_dir):
    """Create histogram showing error distribution."""
    errors = data["errors"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Full histogram
    ax = axes[0]
    ax.hist(errors, bins=30, color=MODEL_COLORS[model], alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}m')
    ax.axvline(np.median(errors), color='blue', linestyle=':', label=f'Median: {np.median(errors):.1f}m')
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # CDF
    ax = axes[1]
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cdf, color=MODEL_COLORS[model], linewidth=2)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(10, color='black', linestyle='--', alpha=0.3, label='10m')
    ax.axvline(20, color='black', linestyle=':', alpha=0.3, label='20m')
    
    # Mark percentiles
    for pct in [50, 75, 90, 95]:
        val = np.percentile(errors, pct)
        ax.plot(val, pct, 'ro', markersize=8)
        ax.annotate(f'P{pct}: {val:.1f}m', (val, pct), textcoords="offset points", 
                   xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Cumulative Distribution')
    ax.set_xlim(0, min(100, np.percentile(errors, 99)))
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'{model} - Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model.lower()}_error_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def analyze_experiment(exp_name, exp_path):
    """Run geographic analysis for one experiment."""
    print(f"\n  Analyzing {exp_name}...")
    
    output_dir = exp_path / "geographic_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model in MODELS:
        data = load_experiment_data(exp_path, model)
        
        if data is None:
            print(f"    ⚠ {model}: No data found")
            continue
        
        print(f"    ✓ {model}: {len(data['errors'])} samples")
        
        # Create plots
        create_error_scatter(data, model, output_dir)
        create_prediction_arrows(data, model, output_dir)
        create_error_heatmap(data, model, output_dir)
        best_worst_stats = create_best_worst_analysis(data, model, output_dir)
        create_error_histogram_by_area(data, model, output_dir)
        
        # Collect stats
        errors = data["errors"]
        results[model] = {
            "n_samples": len(errors),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "within_5m": float(np.mean(errors < 5) * 100),
            "within_10m": float(np.mean(errors < 10) * 100),
            "within_20m": float(np.mean(errors < 20) * 100),
            "p75": float(np.percentile(errors, 75)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            **best_worst_stats
        }
    
    # Save summary
    with open(output_dir / "geographic_summary.json", 'w') as f:
        json.dump({
            "experiment": exp_name,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }, f, indent=2)
    
    return results


def create_cross_experiment_geographic(all_results):
    """Create comparison of geographic patterns across experiments."""
    print("\n📊 Creating Cross-Experiment Geographic Comparison...")
    
    output_dir = Path("testing/comprehensive_comparison/geographic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Best model by area across experiments
    # (This would require more detailed grid analysis - simplified version here)
    
    # Compare best/worst stats across experiments
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    exp_names = list(all_results.keys())
    
    # Best predictions comparison
    ax = axes[0, 0]
    for model in MODELS:
        values = []
        valid_exps = []
        for exp in exp_names:
            if model in all_results[exp]:
                values.append(all_results[exp][model].get('best_mean', np.nan))
                valid_exps.append(exp)
        if values:
            ax.plot(range(len(valid_exps)), values, '-o', 
                   color=MODEL_COLORS[model], label=model, markersize=8)
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Best 20 Predictions - Mean Error')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Worst predictions comparison
    ax = axes[0, 1]
    for model in MODELS:
        values = []
        for exp in exp_names:
            if model in all_results[exp]:
                values.append(all_results[exp][model].get('worst_mean', np.nan))
            else:
                values.append(np.nan)
        ax.plot(range(len(exp_names)), values, '-o', 
               color=MODEL_COLORS[model], label=model, markersize=8)
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Worst 20 Predictions - Mean Error')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # P90 comparison
    ax = axes[1, 0]
    for model in MODELS:
        values = []
        for exp in exp_names:
            if model in all_results[exp]:
                values.append(all_results[exp][model].get('p90', np.nan))
            else:
                values.append(np.nan)
        ax.plot(range(len(exp_names)), values, '-o', 
               color=MODEL_COLORS[model], label=model, markersize=8)
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('P90 Error (m)')
    ax.set_title('90th Percentile Error')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Consistency (std / mean)
    ax = axes[1, 1]
    for model in MODELS:
        values = []
        for exp in exp_names:
            if model in all_results[exp]:
                mean = all_results[exp][model].get('mean_error', 1)
                std = all_results[exp][model].get('std_error', 0)
                values.append(std / mean if mean > 0 else np.nan)
            else:
                values.append(np.nan)
        ax.plot(range(len(exp_names)), values, '-o', 
               color=MODEL_COLORS[model], label=model, markersize=8)
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Prediction Consistency (lower = more consistent)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Geographic Analysis Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "cross_experiment_geographic.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ cross_experiment_geographic.png")


def main():
    """Run geographic analysis for all experiments."""
    print("=" * 70)
    print("GEOGRAPHIC ANALYSIS - ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    for exp_name, exp_path in ALL_EXPERIMENTS.items():
        if not exp_path.exists():
            print(f"\n  ✗ {exp_name}: Path not found")
            continue
        
        results = analyze_experiment(exp_name, exp_path)
        if results:
            all_results[exp_name] = results
    
    # Create cross-experiment comparison
    if len(all_results) > 1:
        create_cross_experiment_geographic(all_results)
    
    print("\n" + "=" * 70)
    print("GEOGRAPHIC ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Analyzed {len(all_results)} experiments")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
