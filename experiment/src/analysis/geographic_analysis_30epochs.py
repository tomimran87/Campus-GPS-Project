"""
Geographic Analysis for All Models (30 Epochs)
===============================================
Performs detailed geographic error analysis for all models:
- Best/worst prediction locations
- Grid-based area analysis  
- Spatial error patterns
- Per-model and comparison outputs

Output Structure:
    testing/30epochs/
    ├── ResNet18/geographic/
    ├── EfficientNet/geographic/
    ├── ConvNeXt/geographic/
    └── comparison/geographic/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


# ============================================
# Configuration
# ============================================
BASE_DIR = Path("testing/30epochs")

MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]

COLORS = {
    "ResNet18": "#e41a1c",
    "EfficientNet": "#377eb8", 
    "ConvNeXt": "#4daf4a",
}

SOURCE_FILES = {
    "ResNet18": ("resnet18_errors.npy", "resnet18_predictions.npy", "resnet18_labels.npy"),
    "EfficientNet": ("efficientnet_errors.npy", "efficientnet_predictions.npy", "efficientnet_labels.npy"),
    "ConvNeXt": ("convnext_errors.npy", "convnext_predictions.npy", "convnext_labels.npy"),
}

TOP_N = 20  # Number of best/worst predictions to analyze
GRID_SIZE = 8  # Grid divisions for area analysis


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def setup_directories():
    """Create output directories."""
    for model in MODELS:
        geo_dir = BASE_DIR / model / "geographic"
        geo_dir.mkdir(parents=True, exist_ok=True)
    
    comp_geo_dir = BASE_DIR / "comparison" / "geographic"
    comp_geo_dir.mkdir(parents=True, exist_ok=True)
    
    return comp_geo_dir


def load_model_data():
    """Load all model results."""
    results = {}
    
    for model, (err_file, pred_file, label_file) in SOURCE_FILES.items():
        # Try from organized directory first
        model_dir = BASE_DIR / model
        
        err_path = model_dir / err_file if (model_dir / err_file).exists() else Path(err_file)
        pred_path = model_dir / pred_file if (model_dir / pred_file).exists() else Path(pred_file)
        label_path = model_dir / label_file if (model_dir / label_file).exists() else Path(label_file)
        
        try:
            results[model] = {
                "errors": np.load(err_path),
                "predictions": np.load(pred_path),
                "labels": np.load(label_path),
            }
            print(f"✓ Loaded {model}")
        except FileNotFoundError:
            print(f"✗ {model}: Files not found")
    
    return results


class GeographicAnalyzer:
    """Performs geographic analysis for a single model."""
    
    def __init__(self, model_name, errors, predictions, labels):
        self.model = model_name
        self.errors = errors
        self.predictions = predictions
        self.labels = labels
        self.output_dir = BASE_DIR / model_name / "geographic"
    
    def get_best_worst_predictions(self):
        """Get indices of best and worst predictions."""
        sorted_indices = np.argsort(self.errors)
        best_indices = sorted_indices[:TOP_N]
        worst_indices = sorted_indices[-TOP_N:][::-1]
        
        return best_indices, worst_indices
    
    def create_prediction_table(self, indices, title):
        """Create a table of predictions."""
        rows = []
        for rank, idx in enumerate(indices, 1):
            rows.append({
                "rank": rank,
                "index": int(idx),
                "true_lat": float(self.labels[idx, 0]),
                "true_lon": float(self.labels[idx, 1]),
                "pred_lat": float(self.predictions[idx, 0]),
                "pred_lon": float(self.predictions[idx, 1]),
                "error_m": float(self.errors[idx]),
            })
        return rows
    
    def grid_analysis(self):
        """Analyze errors by geographic grid."""
        lat_min, lat_max = self.labels[:, 0].min(), self.labels[:, 0].max()
        lon_min, lon_max = self.labels[:, 1].min(), self.labels[:, 1].max()
        
        lat_step = (lat_max - lat_min) / GRID_SIZE
        lon_step = (lon_max - lon_min) / GRID_SIZE
        
        grid_errors = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        for label, error in zip(self.labels, self.errors):
            lat_idx = min(int((label[0] - lat_min) / lat_step), GRID_SIZE - 1)
            lon_idx = min(int((label[1] - lon_min) / lon_step), GRID_SIZE - 1)
            grid_errors[lat_idx][lon_idx].append(error)
        
        grid_stats = np.full((GRID_SIZE, GRID_SIZE), np.nan)
        grid_counts = np.zeros((GRID_SIZE, GRID_SIZE))
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if len(grid_errors[i][j]) > 0:
                    grid_stats[i, j] = np.mean(grid_errors[i][j])
                    grid_counts[i, j] = len(grid_errors[i][j])
        
        return grid_stats, grid_counts, (lat_min, lat_max, lon_min, lon_max)
    
    def plot_error_heatmap(self):
        """Plot error heatmap by location."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(self.labels[:, 1], self.labels[:, 0],
                            c=self.errors, cmap='RdYlGn_r',
                            s=60, alpha=0.7, edgecolors='white', linewidth=0.5,
                            vmin=0, vmax=min(50, np.percentile(self.errors, 95)))
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Error (meters)', fontsize=12)
        
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title(f"{self.model} - Geographic Error Heatmap\n(Mean: {np.mean(self.errors):.2f}m)", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_error_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_grid_heatmap(self):
        """Plot grid-based mean error heatmap."""
        grid_stats, grid_counts, bounds = self.grid_analysis()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Mean error heatmap
        im1 = axes[0].imshow(grid_stats, cmap='RdYlGn_r', origin='lower', 
                            extent=[bounds[2], bounds[3], bounds[0], bounds[1]],
                            vmin=0, vmax=np.nanmax(grid_stats))
        axes[0].set_xlabel("Longitude", fontsize=12)
        axes[0].set_ylabel("Latitude", fontsize=12)
        axes[0].set_title(f"{self.model} - Grid Mean Error", fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Mean Error (m)', fontsize=11)
        
        # Sample count heatmap
        im2 = axes[1].imshow(grid_counts, cmap='Blues', origin='lower',
                            extent=[bounds[2], bounds[3], bounds[0], bounds[1]])
        axes[1].set_xlabel("Longitude", fontsize=12)
        axes[1].set_ylabel("Latitude", fontsize=12)
        axes[1].set_title(f"{self.model} - Sample Density", fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Number of Samples', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_grid_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return grid_stats, grid_counts
    
    def plot_best_worst_locations(self):
        """Plot best and worst prediction locations."""
        best_idx, worst_idx = self.get_best_worst_predictions()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all points (gray)
        ax.scatter(self.labels[:, 1], self.labels[:, 0], 
                  c='lightgray', s=20, alpha=0.3, label='All samples')
        
        # Plot best (green)
        ax.scatter(self.labels[best_idx, 1], self.labels[best_idx, 0],
                  c='green', s=100, alpha=0.8, edgecolors='white', linewidth=1,
                  label=f'Best {TOP_N} (error < {self.errors[best_idx[-1]]:.1f}m)')
        
        # Plot worst (red)
        ax.scatter(self.labels[worst_idx, 1], self.labels[worst_idx, 0],
                  c='red', s=100, alpha=0.8, edgecolors='white', linewidth=1,
                  label=f'Worst {TOP_N} (error > {self.errors[worst_idx[-1]]:.1f}m)')
        
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title(f"{self.model} - Best/Worst Prediction Locations", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_best_worst_locations.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_arrows(self):
        """Plot arrows from true to predicted locations."""
        fig, ax = plt.subplots(figsize=(14, 11))
        
        # Sample to avoid too many arrows
        n_samples = min(200, len(self.errors))
        indices = np.random.choice(len(self.errors), n_samples, replace=False)
        
        for idx in indices:
            error = self.errors[idx]
            color = 'green' if error < 10 else ('orange' if error < 30 else 'red')
            ax.annotate('',
                       xy=(self.predictions[idx, 1], self.predictions[idx, 0]),
                       xytext=(self.labels[idx, 1], self.labels[idx, 0]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=0.8))
        
        ax.scatter(self.labels[:, 1], self.labels[:, 0],
                  c='blue', s=15, alpha=0.5, label='True locations', zorder=5)
        
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title(f"{self.model} - Prediction Vectors\n(Green<10m, Orange<30m, Red>30m)", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_prediction_arrows.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_error_by_coordinate(self):
        """Plot error vs latitude/longitude."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Error vs Latitude
        axes[0].scatter(self.labels[:, 0], self.errors, 
                       c=COLORS[self.model], alpha=0.5, s=20)
        axes[0].set_xlabel("Latitude", fontsize=12)
        axes[0].set_ylabel("Error (meters)", fontsize=12)
        axes[0].set_title(f"{self.model} - Error vs Latitude", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Error vs Longitude
        axes[1].scatter(self.labels[:, 1], self.errors,
                       c=COLORS[self.model], alpha=0.5, s=20)
        axes[1].set_xlabel("Longitude", fontsize=12)
        axes[1].set_ylabel("Error (meters)", fontsize=12)
        axes[1].set_title(f"{self.model} - Error vs Longitude", fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_error_by_coordinate.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_analysis_report(self):
        """Save detailed analysis report."""
        best_idx, worst_idx = self.get_best_worst_predictions()
        best_table = self.create_prediction_table(best_idx, "Best")
        worst_table = self.create_prediction_table(worst_idx, "Worst")
        
        report = {
            "model": self.model,
            "total_samples": len(self.errors),
            "statistics": {
                "mean_error": float(np.mean(self.errors)),
                "median_error": float(np.median(self.errors)),
                "std_error": float(np.std(self.errors)),
                "min_error": float(np.min(self.errors)),
                "max_error": float(np.max(self.errors)),
            },
            "best_predictions": best_table,
            "worst_predictions": worst_table,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "geographic_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Also save as readable text
        with open(self.output_dir / "geographic_analysis.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"GEOGRAPHIC ANALYSIS: {self.model}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total test samples: {len(self.errors)}\n")
            f.write(f"Mean error: {np.mean(self.errors):.2f}m\n")
            f.write(f"Median error: {np.median(self.errors):.2f}m\n\n")
            
            f.write("-" * 70 + "\n")
            f.write(f"TOP {TOP_N} BEST PREDICTIONS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Rank':<6}{'Index':<8}{'True Lat':<14}{'True Lon':<14}{'Error (m)':<10}\n")
            for row in best_table:
                f.write(f"{row['rank']:<6}{row['index']:<8}{row['true_lat']:<14.6f}{row['true_lon']:<14.6f}{row['error_m']:<10.2f}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"TOP {TOP_N} WORST PREDICTIONS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Rank':<6}{'Index':<8}{'True Lat':<14}{'True Lon':<14}{'Error (m)':<10}\n")
            for row in worst_table:
                f.write(f"{row['rank']:<6}{row['index']:<8}{row['true_lat']:<14.6f}{row['true_lon']:<14.6f}{row['error_m']:<10.2f}\n")
    
    def run_full_analysis(self):
        """Run all geographic analyses."""
        print(f"\n  Analyzing {self.model}...")
        
        self.plot_error_heatmap()
        print(f"    ✓ 01_error_heatmap.png")
        
        self.plot_grid_heatmap()
        print(f"    ✓ 02_grid_analysis.png")
        
        self.plot_best_worst_locations()
        print(f"    ✓ 03_best_worst_locations.png")
        
        self.plot_prediction_arrows()
        print(f"    ✓ 04_prediction_arrows.png")
        
        self.plot_error_by_coordinate()
        print(f"    ✓ 05_error_by_coordinate.png")
        
        self.save_analysis_report()
        print(f"    ✓ geographic_analysis.json/txt")


def generate_comparison_geographic(all_results):
    """Generate geographic comparison plots across all models."""
    output_dir = BASE_DIR / "comparison" / "geographic"
    print("\n  Generating geographic comparison...")
    
    # Side-by-side error heatmaps
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model, data) in zip(axes, all_results.items()):
        scatter = ax.scatter(data["labels"][:, 1], data["labels"][:, 0],
                            c=data["errors"], cmap='RdYlGn_r',
                            s=30, alpha=0.7, edgecolors='white', linewidth=0.3,
                            vmin=0, vmax=50)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        mean_err = np.mean(data["errors"])
        ax.set_title(f"{model}\n(Mean: {mean_err:.1f}m)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Error (meters)', fontsize=11)
    
    plt.suptitle("Geographic Error Comparison - All Models (30 Epochs)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "01_geographic_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ 01_geographic_comparison.png")
    
    # Overlay best/worst from all models
    fig, ax = plt.subplots(figsize=(14, 11))
    
    for model, data in all_results.items():
        sorted_idx = np.argsort(data["errors"])
        best_idx = sorted_idx[:10]
        worst_idx = sorted_idx[-10:]
        
        ax.scatter(data["labels"][best_idx, 1], data["labels"][best_idx, 0],
                  c=COLORS[model], s=100, alpha=0.7, marker='o',
                  label=f'{model} Best 10', edgecolors='white')
        ax.scatter(data["labels"][worst_idx, 1], data["labels"][worst_idx, 0],
                  c=COLORS[model], s=100, alpha=0.7, marker='X',
                  label=f'{model} Worst 10', edgecolors='black')
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Best/Worst Locations - All Models (○=Best, ✕=Worst)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_best_worst_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ 02_best_worst_all_models.png")
    
    # Error by coordinate comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for model, data in all_results.items():
        axes[0, 0].scatter(data["labels"][:, 0], data["errors"],
                          c=COLORS[model], alpha=0.3, s=10, label=model)
        axes[0, 1].scatter(data["labels"][:, 1], data["errors"],
                          c=COLORS[model], alpha=0.3, s=10, label=model)
    
    axes[0, 0].set_xlabel("Latitude", fontsize=11)
    axes[0, 0].set_ylabel("Error (meters)", fontsize=11)
    axes[0, 0].set_title("Error vs Latitude - All Models", fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("Longitude", fontsize=11)
    axes[0, 1].set_ylabel("Error (meters)", fontsize=11)
    axes[0, 1].set_title("Error vs Longitude - All Models", fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean error by latitude bins
    lat_bins = np.linspace(
        min(d["labels"][:, 0].min() for d in all_results.values()),
        max(d["labels"][:, 0].max() for d in all_results.values()),
        10
    )
    
    for model, data in all_results.items():
        bin_means = []
        bin_centers = []
        for i in range(len(lat_bins) - 1):
            mask = (data["labels"][:, 0] >= lat_bins[i]) & (data["labels"][:, 0] < lat_bins[i+1])
            if mask.sum() > 0:
                bin_means.append(np.mean(data["errors"][mask]))
                bin_centers.append((lat_bins[i] + lat_bins[i+1]) / 2)
        axes[1, 0].plot(bin_centers, bin_means, '-o', color=COLORS[model], label=model, linewidth=2)
    
    axes[1, 0].set_xlabel("Latitude", fontsize=11)
    axes[1, 0].set_ylabel("Mean Error (meters)", fontsize=11)
    axes[1, 0].set_title("Mean Error by Latitude Region", fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean error by longitude bins
    lon_bins = np.linspace(
        min(d["labels"][:, 1].min() for d in all_results.values()),
        max(d["labels"][:, 1].max() for d in all_results.values()),
        10
    )
    
    for model, data in all_results.items():
        bin_means = []
        bin_centers = []
        for i in range(len(lon_bins) - 1):
            mask = (data["labels"][:, 1] >= lon_bins[i]) & (data["labels"][:, 1] < lon_bins[i+1])
            if mask.sum() > 0:
                bin_means.append(np.mean(data["errors"][mask]))
                bin_centers.append((lon_bins[i] + lon_bins[i+1]) / 2)
        axes[1, 1].plot(bin_centers, bin_means, '-o', color=COLORS[model], label=model, linewidth=2)
    
    axes[1, 1].set_xlabel("Longitude", fontsize=11)
    axes[1, 1].set_ylabel("Mean Error (meters)", fontsize=11)
    axes[1, 1].set_title("Mean Error by Longitude Region", fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Geographic Error Analysis - All Models (30 Epochs)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_by_coordinate_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ 03_error_by_coordinate_comparison.png")


def main():
    print("=" * 70)
    print("GEOGRAPHIC ANALYSIS - ALL MODELS (30 EPOCHS)")
    print("=" * 70)
    
    setup_directories()
    all_results = load_model_data()
    
    if len(all_results) == 0:
        print("\n✗ No model results found. Run training first.")
        return
    
    print("\nGenerating individual model analyses...")
    
    for model, data in all_results.items():
        analyzer = GeographicAnalyzer(
            model, 
            data["errors"], 
            data["predictions"], 
            data["labels"]
        )
        analyzer.run_full_analysis()
    
    generate_comparison_geographic(all_results)
    
    print("\n" + "=" * 70)
    print("✓ GEOGRAPHIC ANALYSIS COMPLETE!")
    print(f"  Output: {BASE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
