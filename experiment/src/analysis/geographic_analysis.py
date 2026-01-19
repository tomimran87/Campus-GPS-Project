"""
Geographic Error Analysis for GPS Localization
================================================
Analyzes prediction errors by geographic location to identify:
- Areas with best prediction accuracy
- Areas with worst prediction accuracy
- Spatial patterns in error distribution

Outputs:
- Log file with best/worst locations
- Scatter plot of errors by location
- Data for creating heatmaps in external tools

Usage:
    python geographic_analysis.py

Output Files:
    - geographic_results/best_predictions.csv
    - geographic_results/worst_predictions.csv
    - geographic_results/all_predictions.csv
    - geographic_results/geographic_analysis.log
    - geographic_results/plots/error_scatter.png
    - geographic_results/plots/error_by_location.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from data_loader import GPSDataManager
from models import EfficientNetGPS
from loss import HaversineLoss


# ============================================
# Configuration
# ============================================
CONFIG = {
    "X_PATH": "/home/liranatt/project/main_project/Latest_data/Latest_data/X.npy",
    "Y_PATH": "/home/liranatt/project/main_project/Latest_data/Latest_data/y.npy",
    "MODEL_PATH": "EfficientNet_gps.pth",
    "MIN_VAL_PATH": "min_val.npy",
    "MAX_VAL_PATH": "max_val.npy",
    "BATCH_SIZE": 32,
    "OUTPUT_DIR": "geographic_results",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "TOP_N_BEST": 50,      # Number of best predictions to log
    "TOP_N_WORST": 50,     # Number of worst predictions to log
    "GRID_SIZE": 10,       # Grid size for area-based analysis
}


# ============================================
# Utility Functions
# ============================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two GPS points in meters."""
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    return R * c


def setup_output_dirs():
    """Create output directories."""
    base_dir = Path(CONFIG["OUTPUT_DIR"])
    dirs = {
        "base": base_dir,
        "plots": base_dir / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


class Logger:
    """Simple logger for file and console output."""
    def __init__(self, path):
        self.file = open(path, "w")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)
        self.file.write(full_msg + "\n")
        self.file.flush()
    
    def close(self):
        self.file.close()


# ============================================
# Main Analysis Functions
# ============================================
def load_model_and_data():
    """Load the trained model and test data."""
    # Load model
    model = EfficientNetGPS()
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"]))
    model = model.to(CONFIG["DEVICE"])
    model.eval()
    
    # Load normalization parameters
    min_val = np.load(CONFIG["MIN_VAL_PATH"])
    max_val = np.load(CONFIG["MAX_VAL_PATH"])
    
    # Load data
    data_manager = GPSDataManager(CONFIG["X_PATH"], CONFIG["Y_PATH"], CONFIG["BATCH_SIZE"])
    train_loader, val_loader, test_loader = data_manager.get_loaders()
    
    return model, test_loader, min_val, max_val


def get_all_predictions(model, test_loader, min_val, max_val):
    """Get predictions for all test samples."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_errors = []
    sample_indices = []
    
    idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(CONFIG["DEVICE"])
            outputs = model(images)
            
            # Denormalize
            preds_real = outputs.cpu().numpy() * (max_val - min_val) + min_val
            labels_real = labels.numpy() * (max_val - min_val) + min_val
            
            for pred, label in zip(preds_real, labels_real):
                error = haversine_distance(pred[0], pred[1], label[0], label[1])
                all_predictions.append(pred)
                all_labels.append(label)
                all_errors.append(error)
                sample_indices.append(idx)
                idx += 1
    
    return {
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "errors": np.array(all_errors),
        "indices": np.array(sample_indices),
    }


def analyze_by_location(data, logger, dirs):
    """Analyze errors by geographic location."""
    predictions = data["predictions"]
    labels = data["labels"]
    errors = data["errors"]
    indices = data["indices"]
    
    # Create combined data array
    combined = []
    for i in range(len(errors)):
        combined.append({
            "index": int(indices[i]),
            "true_lat": float(labels[i, 0]),
            "true_lon": float(labels[i, 1]),
            "pred_lat": float(predictions[i, 0]),
            "pred_lon": float(predictions[i, 1]),
            "error_m": float(errors[i]),
        })
    
    # Sort by error
    combined.sort(key=lambda x: x["error_m"])
    
    # Best predictions
    best = combined[:CONFIG["TOP_N_BEST"]]
    worst = combined[-CONFIG["TOP_N_WORST"]:][::-1]  # Reverse to have worst first
    
    # Log results
    logger.log("=" * 70)
    logger.log("GEOGRAPHIC ERROR ANALYSIS")
    logger.log("=" * 70)
    logger.log(f"Total test samples: {len(errors)}")
    logger.log(f"Mean error: {np.mean(errors):.2f}m")
    logger.log(f"Median error: {np.median(errors):.2f}m")
    logger.log("")
    
    # Best predictions
    logger.log("=" * 70)
    logger.log(f"TOP {CONFIG['TOP_N_BEST']} BEST PREDICTIONS (Lowest Error)")
    logger.log("=" * 70)
    logger.log(f"{'Rank':<6} {'Index':<8} {'True Lat':<12} {'True Lon':<12} {'Pred Lat':<12} {'Pred Lon':<12} {'Error (m)':<10}")
    logger.log("-" * 70)
    
    for rank, item in enumerate(best, 1):
        logger.log(
            f"{rank:<6} {item['index']:<8} {item['true_lat']:<12.6f} {item['true_lon']:<12.6f} "
            f"{item['pred_lat']:<12.6f} {item['pred_lon']:<12.6f} {item['error_m']:<10.2f}"
        )
    
    # Worst predictions
    logger.log("")
    logger.log("=" * 70)
    logger.log(f"TOP {CONFIG['TOP_N_WORST']} WORST PREDICTIONS (Highest Error)")
    logger.log("=" * 70)
    logger.log(f"{'Rank':<6} {'Index':<8} {'True Lat':<12} {'True Lon':<12} {'Pred Lat':<12} {'Pred Lon':<12} {'Error (m)':<10}")
    logger.log("-" * 70)
    
    for rank, item in enumerate(worst, 1):
        logger.log(
            f"{rank:<6} {item['index']:<8} {item['true_lat']:<12.6f} {item['true_lon']:<12.6f} "
            f"{item['pred_lat']:<12.6f} {item['pred_lon']:<12.6f} {item['error_m']:<10.2f}"
        )
    
    # Save to CSV files
    save_to_csv(best, dirs["base"] / "best_predictions.csv")
    save_to_csv(worst, dirs["base"] / "worst_predictions.csv")
    save_to_csv(combined, dirs["base"] / "all_predictions.csv")
    
    logger.log("")
    logger.log(f"✓ Saved best predictions to: {dirs['base'] / 'best_predictions.csv'}")
    logger.log(f"✓ Saved worst predictions to: {dirs['base'] / 'worst_predictions.csv'}")
    logger.log(f"✓ Saved all predictions to: {dirs['base'] / 'all_predictions.csv'}")
    
    return best, worst, combined


def analyze_by_grid(data, logger, dirs):
    """Analyze errors by dividing the area into a grid."""
    labels = data["labels"]
    errors = data["errors"]
    
    # Get bounds
    lat_min, lat_max = labels[:, 0].min(), labels[:, 0].max()
    lon_min, lon_max = labels[:, 1].min(), labels[:, 1].max()
    
    grid_size = CONFIG["GRID_SIZE"]
    lat_step = (lat_max - lat_min) / grid_size
    lon_step = (lon_max - lon_min) / grid_size
    
    # Create grid
    grid_errors = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    
    for label, error in zip(labels, errors):
        lat_idx = min(int((label[0] - lat_min) / lat_step), grid_size - 1)
        lon_idx = min(int((label[1] - lon_min) / lon_step), grid_size - 1)
        grid_errors[lat_idx][lon_idx].append(error)
    
    # Compute statistics for each cell
    logger.log("")
    logger.log("=" * 70)
    logger.log(f"GRID-BASED ANALYSIS ({grid_size}x{grid_size} grid)")
    logger.log("=" * 70)
    
    grid_stats = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell_errors = grid_errors[i][j]
            if len(cell_errors) > 0:
                lat_center = lat_min + (i + 0.5) * lat_step
                lon_center = lon_min + (j + 0.5) * lon_step
                mean_error = np.mean(cell_errors)
                grid_stats.append({
                    "lat_center": lat_center,
                    "lon_center": lon_center,
                    "mean_error": mean_error,
                    "count": len(cell_errors),
                    "grid_i": i,
                    "grid_j": j,
                })
    
    # Sort by mean error
    grid_stats.sort(key=lambda x: x["mean_error"])
    
    # Best areas
    logger.log("")
    logger.log("TOP 5 BEST AREAS (Lowest Mean Error):")
    logger.log(f"{'Lat Center':<15} {'Lon Center':<15} {'Mean Error (m)':<15} {'Samples':<10}")
    logger.log("-" * 55)
    for item in grid_stats[:5]:
        logger.log(
            f"{item['lat_center']:<15.6f} {item['lon_center']:<15.6f} "
            f"{item['mean_error']:<15.2f} {item['count']:<10}"
        )
    
    # Worst areas
    logger.log("")
    logger.log("TOP 5 WORST AREAS (Highest Mean Error):")
    logger.log(f"{'Lat Center':<15} {'Lon Center':<15} {'Mean Error (m)':<15} {'Samples':<10}")
    logger.log("-" * 55)
    for item in grid_stats[-5:][::-1]:
        logger.log(
            f"{item['lat_center']:<15.6f} {item['lon_center']:<15.6f} "
            f"{item['mean_error']:<15.2f} {item['count']:<10}"
        )
    
    # Save grid statistics
    with open(dirs["base"] / "grid_analysis.json", "w") as f:
        json.dump(grid_stats, f, indent=2)
    
    logger.log(f"\n✓ Saved grid analysis to: {dirs['base'] / 'grid_analysis.json'}")
    
    return grid_stats


def save_to_csv(data, filepath):
    """Save prediction data to CSV file."""
    with open(filepath, "w") as f:
        # Header
        f.write("index,true_lat,true_lon,pred_lat,pred_lon,error_m\n")
        
        # Data rows
        for item in data:
            f.write(
                f"{item['index']},{item['true_lat']:.6f},{item['true_lon']:.6f},"
                f"{item['pred_lat']:.6f},{item['pred_lon']:.6f},{item['error_m']:.2f}\n"
            )


# ============================================
# Plotting Functions
# ============================================
def plot_error_scatter(data, best, worst, dirs):
    """Create scatter plot of errors by location."""
    labels = data["labels"]
    errors = data["errors"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. All samples colored by error
    ax = axes[0]
    scatter = ax.scatter(
        labels[:, 1], labels[:, 0],  # lon, lat
        c=errors, cmap='RdYlGn_r', s=30, alpha=0.7,
        vmin=0, vmax=np.percentile(errors, 95)  # Cap at 95th percentile for visibility
    )
    plt.colorbar(scatter, ax=ax, label='Error (meters)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Prediction Errors by Location')
    ax.grid(True, alpha=0.3)
    
    # 2. Best and worst highlighted
    ax = axes[1]
    
    # Plot all samples in gray
    ax.scatter(labels[:, 1], labels[:, 0], c='lightgray', s=20, alpha=0.3, label='All samples')
    
    # Plot best in green
    best_lats = [b['true_lat'] for b in best]
    best_lons = [b['true_lon'] for b in best]
    ax.scatter(best_lons, best_lats, c='green', s=50, alpha=0.7, 
               label=f'Best {len(best)} (< {best[-1]["error_m"]:.1f}m)')
    
    # Plot worst in red
    worst_lats = [w['true_lat'] for w in worst]
    worst_lons = [w['true_lon'] for w in worst]
    ax.scatter(worst_lons, worst_lats, c='red', s=50, alpha=0.7,
               label=f'Worst {len(worst)} (> {worst[-1]["error_m"]:.1f}m)')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Best vs Worst Prediction Locations')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Geographic Error Analysis - BGU Campus', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "error_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_heatmap(grid_stats, dirs):
    """Create heatmap of errors by grid cell."""
    grid_size = CONFIG["GRID_SIZE"]
    
    # Create grid array
    error_grid = np.full((grid_size, grid_size), np.nan)
    count_grid = np.zeros((grid_size, grid_size))
    
    for item in grid_stats:
        i, j = item['grid_i'], item['grid_j']
        error_grid[i, j] = item['mean_error']
        count_grid[i, j] = item['count']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Mean error heatmap
    ax = axes[0]
    im = ax.imshow(error_grid, cmap='RdYlGn_r', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Error (meters)')
    ax.set_xlabel('Longitude Grid Index')
    ax.set_ylabel('Latitude Grid Index')
    ax.set_title('Mean Error by Grid Cell')
    
    # Add text annotations
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(error_grid[i, j]):
                ax.text(j, i, f'{error_grid[i, j]:.0f}', ha='center', va='center', 
                       fontsize=8, color='black' if error_grid[i, j] < 15 else 'white')
    
    # 2. Sample count heatmap
    ax = axes[1]
    im = ax.imshow(count_grid, cmap='Blues', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Sample Count')
    ax.set_xlabel('Longitude Grid Index')
    ax.set_ylabel('Latitude Grid Index')
    ax.set_title('Sample Count by Grid Cell')
    
    # Add text annotations
    for i in range(grid_size):
        for j in range(grid_size):
            if count_grid[i, j] > 0:
                ax.text(j, i, f'{int(count_grid[i, j])}', ha='center', va='center',
                       fontsize=8, color='black' if count_grid[i, j] < 30 else 'white')
    
    plt.suptitle('Grid-Based Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "error_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_vectors(data, dirs):
    """Plot vectors from true to predicted locations."""
    predictions = data["predictions"]
    labels = data["labels"]
    errors = data["errors"]
    
    # Sort by error and select samples
    sorted_indices = np.argsort(errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Best predictions
    ax = axes[0]
    best_indices = sorted_indices[:50]
    for idx in best_indices:
        ax.arrow(
            labels[idx, 1], labels[idx, 0],  # lon, lat (start)
            predictions[idx, 1] - labels[idx, 1],  # dlon
            predictions[idx, 0] - labels[idx, 0],  # dlat
            head_width=0.00005, head_length=0.00003,
            fc='green', ec='green', alpha=0.5
        )
    ax.scatter(labels[best_indices, 1], labels[best_indices, 0], c='blue', s=10, zorder=5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Best 50 Predictions (Arrows: True → Predicted)')
    ax.grid(True, alpha=0.3)
    
    # 2. Worst predictions
    ax = axes[1]
    worst_indices = sorted_indices[-50:]
    for idx in worst_indices:
        ax.arrow(
            labels[idx, 1], labels[idx, 0],  # lon, lat (start)
            predictions[idx, 1] - labels[idx, 1],  # dlon
            predictions[idx, 0] - labels[idx, 0],  # dlat
            head_width=0.00015, head_length=0.0001,
            fc='red', ec='red', alpha=0.5
        )
    ax.scatter(labels[worst_indices, 1], labels[worst_indices, 0], c='blue', s=10, zorder=5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Worst 50 Predictions (Arrows: True → Predicted)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Error Vectors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "prediction_vectors.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_google_maps_data(best, worst, dirs):
    """Create a file with Google Maps links for easy visualization."""
    with open(dirs["base"] / "google_maps_links.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GOOGLE MAPS LINKS FOR BEST/WORST PREDICTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TOP 10 BEST PREDICTIONS (Lowest Error):\n")
        f.write("-" * 80 + "\n")
        for i, item in enumerate(best[:10], 1):
            f.write(f"{i}. Error: {item['error_m']:.2f}m\n")
            f.write(f"   True:      https://www.google.com/maps?q={item['true_lat']},{item['true_lon']}\n")
            f.write(f"   Predicted: https://www.google.com/maps?q={item['pred_lat']},{item['pred_lon']}\n\n")
        
        f.write("\nTOP 10 WORST PREDICTIONS (Highest Error):\n")
        f.write("-" * 80 + "\n")
        for i, item in enumerate(worst[:10], 1):
            f.write(f"{i}. Error: {item['error_m']:.2f}m\n")
            f.write(f"   True:      https://www.google.com/maps?q={item['true_lat']},{item['true_lon']}\n")
            f.write(f"   Predicted: https://www.google.com/maps?q={item['pred_lat']},{item['pred_lon']}\n\n")
    
    print(f"✓ Saved Google Maps links to: {dirs['base'] / 'google_maps_links.txt'}")


# ============================================
# Main Execution
# ============================================
def main():
    """Run geographic analysis."""
    print("=" * 60)
    print("GEOGRAPHIC ERROR ANALYSIS")
    print("=" * 60)
    print(f"Device: {CONFIG['DEVICE']}")
    print()
    
    # Setup
    dirs = setup_output_dirs()
    logger = Logger(dirs["base"] / "geographic_analysis.log")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load model and data
        logger.log("Loading model and data...")
        model, test_loader, min_val, max_val = load_model_and_data()
        logger.log(f"Model loaded from: {CONFIG['MODEL_PATH']}")
        logger.log(f"Test samples: {len(test_loader.dataset)}")
        
        # Get all predictions
        logger.log("Computing predictions...")
        data = get_all_predictions(model, test_loader, min_val, max_val)
        
        # Analyze by location
        best, worst, all_predictions = analyze_by_location(data, logger, dirs)
        
        # Analyze by grid
        grid_stats = analyze_by_grid(data, logger, dirs)
        
        # Create plots
        logger.log("\nGenerating plots...")
        plot_error_scatter(data, best, worst, dirs)
        plot_error_heatmap(grid_stats, dirs)
        plot_prediction_vectors(data, dirs)
        
        # Create Google Maps links
        create_google_maps_data(best, worst, dirs)
        
        logger.log("")
        logger.log("=" * 70)
        logger.log("ANALYSIS COMPLETE")
        logger.log("=" * 70)
        logger.log(f"Output directory: {dirs['base']}")
        logger.log(f"Plots saved to: {dirs['plots']}")
        logger.log("")
        logger.log("Files created:")
        logger.log("  - best_predictions.csv: Locations with best predictions")
        logger.log("  - worst_predictions.csv: Locations with worst predictions")
        logger.log("  - all_predictions.csv: All predictions for scatter mapping")
        logger.log("  - grid_analysis.json: Grid-based error statistics")
        logger.log("  - google_maps_links.txt: Quick links for visualization")
        logger.log("  - plots/error_scatter.png: Error distribution map")
        logger.log("  - plots/error_heatmap.png: Grid-based error heatmap")
        logger.log("  - plots/prediction_vectors.png: True → Predicted vectors")
        
    except FileNotFoundError as e:
        logger.log(f"ERROR: File not found - {e}")
        logger.log("Make sure you have trained a model first by running main.py")
        raise
    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise
    finally:
        logger.close()
    
    print("\n✓ Analysis complete!")
    print(f"✓ Results saved to: {dirs['base']}")


if __name__ == "__main__":
    main()
