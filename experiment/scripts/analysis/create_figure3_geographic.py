#!/usr/bin/env python3
"""
Geographic Error Visualization on BGU Campus Map

Creates two figures matching the paper specification:
1. (a) Spatial Error Heatmap - showing high-error zones in red
2. (b) Prediction Arrows - showing vector field of prediction displacement

Both visualizations are overlaid on the actual BGU campus map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MAP_PATH = BASE_DIR.parent / "bgu_map.png"
DATA_DIR = Path("/home/liranatt/project/main_project/Latest_data/Latest_data")
RESULTS_DIR = BASE_DIR / "results" / "testing" / "100ep_lr001_fulldata" / "EfficientNet"
OUTPUT_DIR = BASE_DIR / "results" / "paper_figures"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load predictions, labels, and errors from best experiment."""
    predictions = np.load(RESULTS_DIR / "efficientnet_predictions.npy")
    labels = np.load(RESULTS_DIR / "efficientnet_labels.npy")
    errors = np.load(RESULTS_DIR / "efficientnet_errors.npy")
    
    # Load min/max values for denormalization
    min_val = np.load(RESULTS_DIR.parent / "min_val.npy")
    max_val = np.load(RESULTS_DIR.parent / "max_val.npy")
    
    # Check if data is already in real coordinates or normalized
    # If labels are already in GPS range (31.x, 34.x), they're real coords
    if labels.min() > 1:  # Already real coordinates
        predictions_real = predictions
        labels_real = labels
        print("Data is already in real GPS coordinates")
    else:  # Normalized [0,1] - need to denormalize
        predictions_real = predictions * (max_val - min_val) + min_val
        labels_real = labels * (max_val - min_val) + min_val
        print("Denormalized data from [0,1] range")
    
    print(f"Loaded {len(predictions)} samples")
    print(f"Labels range: lat [{labels_real[:,0].min():.6f}, {labels_real[:,0].max():.6f}]")
    print(f"Labels range: lon [{labels_real[:,1].min():.6f}, {labels_real[:,1].max():.6f}]")
    print(f"Errors range: [{errors.min():.2f}m, {errors.max():.2f}m]")
    
    return predictions_real, labels_real, errors, min_val, max_val


def get_map_bounds():
    """
    Define the GPS bounds that correspond to the map image.
    These are calibrated to match the BGU campus map extent.
    
    From the map image (bgu_map.png), this shows the area around:
    - Kreitman Plaza (כיכר קרייטמן)
    - Cummings Plaza (כיכר קאמינגס)
    - Buildings 22, 25, 26, 28, 29, 32, 33, 35, 37
    
    Based on the data:
    - Latitude: 31.261283 to 31.262767  (data range)
    - Longitude: 34.801083 to 34.804469 (data range)
    
    Calibrated map bounds to match visible area:
    """
    # Calibrated to match the BGU campus map visible area
    # Adjusted to ensure data points fall within the map boundaries
    lat_min = 31.2608   # Bottom of map
    lat_max = 31.2632   # Top of map  
    lon_min = 34.7998   # Left of map
    lon_max = 34.8052   # Right of map
    
    return lat_min, lat_max, lon_min, lon_max


def gps_to_pixel(lat, lon, map_width, map_height, bounds):
    """Convert GPS coordinates to pixel coordinates on the map."""
    lat_min, lat_max, lon_min, lon_max = bounds
    
    # Normalize to [0, 1]
    x_norm = (lon - lon_min) / (lon_max - lon_min)
    y_norm = (lat - lat_min) / (lat_max - lat_min)
    
    # Convert to pixels (y is inverted because image origin is top-left)
    x_pixel = x_norm * map_width
    y_pixel = (1 - y_norm) * map_height
    
    return x_pixel, y_pixel


def create_heatmap_figure(labels_real, errors, map_img, bounds, output_path):
    """
    Create Figure 3(a): Spatial Error Heatmap
    Shows each sample point colored by error with strong, visible colors.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Display the map as background
    ax.imshow(map_img, extent=[bounds[2], bounds[3], bounds[0], bounds[1]], 
              aspect='auto', alpha=1.0)
    
    # Sort points by error so high-error points are drawn on top
    sorted_idx = np.argsort(errors)
    labels_sorted = labels_real[sorted_idx]
    errors_sorted = errors[sorted_idx]
    
    # Create a strong, high-contrast colormap
    colors = ['#00FF00', '#FFFF00', '#FF8000', '#FF0000']  # Bright green -> yellow -> orange -> red
    cmap = LinearSegmentedColormap.from_list('error_cmap', colors)
    
    # Plot all points with strong colors and black edges for visibility
    scatter = ax.scatter(
        labels_sorted[:, 1], labels_sorted[:, 0],
        c=errors_sorted, 
        cmap=cmap, 
        s=60,  # Larger markers
        alpha=0.9,  # Nearly opaque
        edgecolors='black', 
        linewidths=1.0,
        vmin=0, 
        vmax=np.percentile(errors, 90),
        zorder=10
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Error (meters)', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('(a) Spatial Error Heatmap\nGreen = Low Error, Red = High Error', fontsize=14, fontweight='bold')
    
    ax.set_xlim(bounds[2], bounds[3])
    ax.set_ylim(bounds[0], bounds[1])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_arrow_figure(predictions_real, labels_real, errors, map_img, bounds, output_path):
    """
    Create Figure 3(b): Prediction Arrows (Vector Field)
    Shows ~20 representative arrows from true location to predicted location.
    Stratified sampling: select points with various error levels for diversity.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Display the map as background
    ax.imshow(map_img, extent=[bounds[2], bounds[3], bounds[0], bounds[1]], 
              aspect='auto', alpha=1.0)
    
    # Select ~20 representative samples stratified by error
    n_samples = 20
    
    # Sort by error and pick evenly spaced samples (to show range of errors)
    sorted_indices = np.argsort(errors)
    
    # Pick samples evenly distributed across error range
    selected_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n_samples, dtype=int)]
    
    # Also ensure we have some high-error samples for visibility
    top_error_indices = sorted_indices[-5:]  # Top 5 highest errors
    selected_indices = np.unique(np.concatenate([selected_indices, top_error_indices]))
    
    # Filter to selected samples
    sel_pred = predictions_real[selected_indices]
    sel_labels = labels_real[selected_indices]
    sel_errors = errors[selected_indices]
    
    # First, show all data points as faint dots
    ax.scatter(
        labels_real[:, 1], labels_real[:, 0],
        c='gray', s=5, alpha=0.3, label='All samples'
    )
    
    # Calculate displacement vectors for selected points
    dlat = sel_pred[:, 0] - sel_labels[:, 0]
    dlon = sel_pred[:, 1] - sel_labels[:, 1]
    
    # Normalize errors for coloring
    max_error = np.percentile(errors, 95)
    norm_errors = np.clip(sel_errors / max_error, 0, 1)
    
    # Create colormap for arrows
    colors_arrow = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # Green -> Yellow -> Orange -> Red
    cmap_arrow = LinearSegmentedColormap.from_list('arrow_cmap', colors_arrow)
    
    # Draw arrows with quiver for cleaner look
    for i in range(len(sel_labels)):
        color = cmap_arrow(norm_errors[i])
        
        # Scale factor for arrow visibility (exaggerate displacement)
        scale = 3.0  # Make arrows more visible
        
        arrow_dx = dlon[i] * scale
        arrow_dy = dlat[i] * scale
        
        # Draw the arrow
        ax.annotate(
            '',
            xy=(sel_labels[i, 1] + arrow_dx, sel_labels[i, 0] + arrow_dy),
            xytext=(sel_labels[i, 1], sel_labels[i, 0]),
            arrowprops=dict(
                arrowstyle='-|>',
                color=color,
                lw=2.5,
                mutation_scale=15,
                alpha=0.9
            ),
            zorder=10
        )
        
        # Mark the true location with a distinct marker
        ax.scatter(
            sel_labels[i, 1], sel_labels[i, 0],
            c=[color], s=80, marker='o',
            edgecolors='white', linewidths=1.5,
            zorder=11
        )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_arrow, norm=plt.Normalize(0, max_error))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Error (meters)', fontsize=12)
    
    # Add legend
    ax.scatter([], [], c='gray', s=20, alpha=0.5, label=f'All samples (n={len(labels_real)})')
    ax.scatter([], [], c='#e74c3c', s=60, edgecolors='white', linewidths=1, 
               label=f'Selected samples (n={len(sel_labels)})')
    ax.legend(loc='upper left', fontsize=9)
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('(b) Prediction Arrows\nVector Field: Prediction Displacement (3x scaled)', fontsize=14, fontweight='bold')
    
    ax.set_xlim(bounds[2], bounds[3])
    ax.set_ylim(bounds[0], bounds[1])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_combined_figure(predictions_real, labels_real, errors, map_img, bounds, output_path):
    """
    Create combined Figure 3 with both (a) and (b) side by side.
    This matches the paper layout exactly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==================== (a) Spatial Error Heatmap ====================
    ax = axes[0]
    ax.imshow(map_img, extent=[bounds[2], bounds[3], bounds[0], bounds[1]], 
              aspect='auto', alpha=1.0)
    
    # Sort points by error so high-error points are drawn on top
    sorted_idx = np.argsort(errors)
    labels_sorted = labels_real[sorted_idx]
    errors_sorted = errors[sorted_idx]
    
    # Strong, high-contrast colormap
    colors_heat = ['#00FF00', '#FFFF00', '#FF8000', '#FF0000']  # Bright green -> yellow -> orange -> red
    cmap_heat = LinearSegmentedColormap.from_list('error_cmap', colors_heat)
    
    # Plot all points with strong colors
    scatter = ax.scatter(
        labels_sorted[:, 1], labels_sorted[:, 0],
        c=errors_sorted, 
        cmap=cmap_heat, 
        s=45,  # Slightly smaller for combined view
        alpha=0.9,
        edgecolors='black', 
        linewidths=0.8,
        vmin=0, 
        vmax=np.percentile(errors, 90),
        zorder=10
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Error (m)', fontsize=10)
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('(a) Spatial Error Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlim(bounds[2], bounds[3])
    ax.set_ylim(bounds[0], bounds[1])
    
    # ==================== (b) Prediction Arrows ====================
    ax = axes[1]
    ax.imshow(map_img, extent=[bounds[2], bounds[3], bounds[0], bounds[1]], 
              aspect='auto', alpha=1.0)
    
    # Select ~20 representative samples
    n_samples = 20
    sorted_indices = np.argsort(errors)
    selected_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n_samples, dtype=int)]
    top_error_indices = sorted_indices[-5:]
    selected_indices = np.unique(np.concatenate([selected_indices, top_error_indices]))
    
    sel_pred = predictions_real[selected_indices]
    sel_labels = labels_real[selected_indices]
    sel_errors = errors[selected_indices]
    
    # Show all points as faint dots
    ax.scatter(labels_real[:, 1], labels_real[:, 0], c='gray', s=4, alpha=0.3)
    
    # Calculate displacements
    dlat = sel_pred[:, 0] - sel_labels[:, 0]
    dlon = sel_pred[:, 1] - sel_labels[:, 1]
    
    max_error = np.percentile(errors, 95)
    norm_errors = np.clip(sel_errors / max_error, 0, 1)
    
    colors_arrow = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    cmap_arrow = LinearSegmentedColormap.from_list('arrow_cmap', colors_arrow)
    
    # Draw arrows
    for i in range(len(sel_labels)):
        color = cmap_arrow(norm_errors[i])
        scale = 3.0
        
        ax.annotate(
            '',
            xy=(sel_labels[i, 1] + dlon[i] * scale, sel_labels[i, 0] + dlat[i] * scale),
            xytext=(sel_labels[i, 1], sel_labels[i, 0]),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=2.0, mutation_scale=12, alpha=0.9),
            zorder=10
        )
        ax.scatter(sel_labels[i, 1], sel_labels[i, 0], c=[color], s=50, marker='o',
                   edgecolors='white', linewidths=1.0, zorder=11)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_arrow, norm=plt.Normalize(0, max_error))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Error (m)', fontsize=10)
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('(b) Prediction Arrows (3× scaled)', fontsize=12, fontweight='bold')
    ax.set_xlim(bounds[2], bounds[3])
    ax.set_ylim(bounds[0], bounds[1])
    
    # Main figure title
    fig.suptitle('Figure 3: Geographic Analysis of Model Failure Modes', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Creating Geographic Error Visualization on BGU Campus Map")
    print("=" * 60)
    
    # Load the campus map
    print(f"\nLoading map from: {MAP_PATH}")
    map_img = Image.open(MAP_PATH)
    map_width, map_height = map_img.size
    print(f"Map dimensions: {map_width} x {map_height}")
    
    # Load prediction data
    print("\nLoading prediction data...")
    predictions_real, labels_real, errors, min_val, max_val = load_data()
    
    # Get map bounds
    bounds = get_map_bounds()
    print(f"\nMap bounds:")
    print(f"  Latitude: {bounds[0]:.6f} to {bounds[1]:.6f}")
    print(f"  Longitude: {bounds[2]:.6f} to {bounds[3]:.6f}")
    
    # Create individual figures
    print("\nGenerating figures...")
    
    create_heatmap_figure(
        labels_real, errors, map_img, bounds,
        OUTPUT_DIR / "figure3a_spatial_error_heatmap.png"
    )
    
    create_arrow_figure(
        predictions_real, labels_real, errors, map_img, bounds,
        OUTPUT_DIR / "figure3b_prediction_arrows.png"
    )
    
    create_combined_figure(
        predictions_real, labels_real, errors, map_img, bounds,
        OUTPUT_DIR / "figure3_geographic_analysis.png"
    )
    
    print("\n" + "=" * 60)
    print("✅ All figures saved to:")
    print(f"   {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
