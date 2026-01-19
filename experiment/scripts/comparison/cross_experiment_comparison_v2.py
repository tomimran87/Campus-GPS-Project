#!/usr/bin/env python3
"""
Cross-Experiment Comparison: All 8 Experiments
================================================

Compares results across ALL experiments:
1. 30 epochs, LR=0.001, Full Data (100%), Regular Test
2. 30 epochs, LR=0.0001, Half Data (50%), Regular Test  
3. 30 epochs, LR=0.0001, Half Data (50%), Augmented Test (UNSEEN locations)
4. 50 epochs, LR=0.001, Full Data (100%), Regular Test
5. 50 epochs, LR=0.0001, Half Data (50%), Regular Test
6. 50 epochs, LR=0.0001, Half Data (50%), Augmented Test (UNSEEN locations)
7. 100 epochs, LR=0.001, Full Data (100%), Regular Test
8. 100 epochs, LR=0.0001, Half Data (50%), Regular Test

Output: testing/cross_experiment_comparison_v2/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


# ============================================
# Configuration
# ============================================
EXPERIMENTS = {
    "Exp1: 30ep Full\n(LR=0.001)": {
        "path": Path("testing/30epochs"),
        "short": "30ep Full",
        "config": "30ep, LR=0.001, 100% data",
        "test_type": "Regular (same distribution)",
    },
    "Exp2: 30ep Half\n(LR=0.0001)": {
        "path": Path("testing/30epochs_lr0.0001_halfdata"),
        "short": "30ep Half",
        "config": "30ep, LR=0.0001, 50% data",
        "test_type": "Regular (same distribution)",
    },
    "Exp3: 30ep Half\n+ Aug Test": {
        "path": Path("testing/30ep_lr0001_halfdata_augtest"),
        "short": "30ep Half+Aug",
        "config": "30ep, LR=0.0001, 50% data, augmented test",
        "test_type": "Augmented (UNSEEN locations)",
    },
    "Exp4: 50ep Full\n(LR=0.001)": {
        "path": Path("testing/50ep_lr001_fulldata"),
        "short": "50ep Full",
        "config": "50ep, LR=0.001, 100% data",
        "test_type": "Regular (same distribution)",
    },
    "Exp5: 50ep Half\n(LR=0.0001)": {
        "path": Path("testing/50ep_lr0001_halfdata"),
        "short": "50ep Half",
        "config": "50ep, LR=0.0001, 50% data",
        "test_type": "Regular (same distribution)",
    },
    "Exp6: 50ep Half\n+ Aug Test": {
        "path": Path("testing/50ep_lr0001_halfdata_augtest"),
        "short": "50ep Half+Aug",
        "config": "50ep, LR=0.0001, 50% data, augmented test",
        "test_type": "Augmented (UNSEEN locations)",
    },
    "Exp7: 100ep Full\n(LR=0.001)": {
        "path": Path("testing/100ep_lr001_fulldata"),
        "short": "100ep Full",
        "config": "100ep, LR=0.001, 100% data",
        "test_type": "Regular (same distribution)",
    },
    "Exp8: 100ep Half\n(LR=0.0001)": {
        "path": Path("testing/100ep_lr0001_halfdata"),
        "short": "100ep Half",
        "config": "100ep, LR=0.0001, 50% data",
        "test_type": "Regular (same distribution)",
    },
}

MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]

# Distinct colors for each experiment
EXP_COLORS = {
    "Exp1: 30ep Full\n(LR=0.001)": "#1f77b4",      # Blue
    "Exp2: 30ep Half\n(LR=0.0001)": "#ff7f0e",     # Orange
    "Exp3: 30ep Half\n+ Aug Test": "#2ca02c",      # Green
    "Exp4: 50ep Full\n(LR=0.001)": "#d62728",      # Red
    "Exp5: 50ep Half\n(LR=0.0001)": "#9467bd",     # Purple
    "Exp6: 50ep Half\n+ Aug Test": "#8c564b",      # Brown
    "Exp7: 100ep Full\n(LR=0.001)": "#e377c2",     # Pink
    "Exp8: 100ep Half\n(LR=0.0001)": "#17becf",    # Cyan
}

MODEL_COLORS = {
    "ResNet18": "#e41a1c",
    "EfficientNet": "#377eb8",
    "ConvNeXt": "#4daf4a",
}

OUTPUT_DIR = Path("testing/cross_experiment_comparison_v2")


def setup_dirs():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)


def load_all_results():
    """Load results from all experiments."""
    all_results = {}
    
    for exp_name, exp_config in EXPERIMENTS.items():
        exp_path = exp_config["path"]
        all_results[exp_name] = {"config": exp_config}
        
        for model in MODELS:
            model_dir = exp_path / model
            
            # Try different file naming patterns
            errors_candidates = [
                model_dir / f"{model.lower()}_errors.npy",
                model_dir / "errors.npy",
            ]
            
            errors_path = None
            for candidate in errors_candidates:
                if candidate.exists():
                    errors_path = candidate
                    break
            
            if errors_path is None:
                # Try root level
                root_candidates = [
                    exp_path / f"{model.lower()}_errors.npy",
                ]
                for candidate in root_candidates:
                    if candidate.exists():
                        errors_path = candidate
                        break
            
            if errors_path is None:
                print(f"⚠ Missing {model} in {exp_name}")
                continue
            
            errors = np.load(errors_path)
            
            # Load predictions and labels
            preds_path = errors_path.parent / errors_path.name.replace("errors", "predictions")
            labels_path = errors_path.parent / errors_path.name.replace("errors", "labels")
            
            preds = np.load(preds_path) if preds_path.exists() else None
            labels = np.load(labels_path) if labels_path.exists() else None
            
            all_results[exp_name][model] = {
                "errors": errors,
                "predictions": preds,
                "labels": labels,
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "std_error": float(np.std(errors)),
                "within_5m": float(np.mean(errors < 5) * 100),
                "within_10m": float(np.mean(errors < 10) * 100),
                "within_20m": float(np.mean(errors < 20) * 100),
                "within_50m": float(np.mean(errors < 50) * 100),
                "max_error": float(np.max(errors)),
                "p95_error": float(np.percentile(errors, 95)),
            }
    
    return all_results


def plot_mean_error_grouped(all_results):
    """Bar chart: Mean error grouped by model, showing all experiments."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    x = np.arange(len(MODELS))
    width = 0.25
    
    for i, exp_name in enumerate(exp_names):
        means = []
        for model in MODELS:
            if model in all_results[exp_name]:
                means.append(all_results[exp_name][model]["mean_error"])
            else:
                means.append(0)
        
        bars = ax.bar(x + i * width, means, width, 
                     label=EXPERIMENTS[exp_name]["short"],
                     color=EXP_COLORS[exp_name], edgecolor='white', alpha=0.8)
        
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title("Mean Error Comparison Across All Experiments", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.legend(fontsize=10, title="Experiment")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "01_mean_error_by_model.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_mean_error_by_model.png")


def plot_accuracy_grouped(all_results):
    """Bar chart: Within 10m accuracy grouped by model."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    x = np.arange(len(MODELS))
    width = 0.25
    
    for i, exp_name in enumerate(exp_names):
        acc = []
        for model in MODELS:
            if model in all_results[exp_name]:
                acc.append(all_results[exp_name][model]["within_10m"])
            else:
                acc.append(0)
        
        bars = ax.bar(x + i * width, acc, width,
                     label=EXPERIMENTS[exp_name]["short"],
                     color=EXP_COLORS[exp_name], edgecolor='white', alpha=0.8)
        
        for bar, val in zip(bars, acc):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy Within 10m (%)", fontsize=12)
    ax.set_title("Accuracy Comparison Across All Experiments", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.legend(fontsize=10, title="Experiment")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "02_accuracy_by_model.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_accuracy_by_model.png")


def plot_cdf_per_model(all_results):
    """CDF comparison per model across experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    for ax, model in zip(axes, MODELS):
        for exp_name in exp_names:
            if model not in all_results[exp_name]:
                continue
            
            errors = all_results[exp_name][model]["errors"]
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
            
            ax.plot(sorted_errors, cdf, color=EXP_COLORS[exp_name], 
                   linewidth=2, label=EXPERIMENTS[exp_name]["short"])
        
        ax.axvline(x=10, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Error (meters)", fontsize=11)
        ax.set_ylabel("Cumulative %", fontsize=11)
        ax.set_title(f"{model}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 105)
    
    plt.suptitle("CDF Comparison Per Model Across Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "03_cdf_per_model.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_cdf_per_model.png")


def plot_cdf_per_experiment(all_results):
    """CDF comparison per experiment (all models in each)."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    n_exp = len(exp_names)
    
    fig, axes = plt.subplots(1, n_exp, figsize=(6*n_exp, 6))
    if n_exp == 1:
        axes = [axes]
    
    for ax, exp_name in zip(axes, exp_names):
        for model in MODELS:
            if model not in all_results[exp_name]:
                continue
            
            errors = all_results[exp_name][model]["errors"]
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
            
            ax.plot(sorted_errors, cdf, color=MODEL_COLORS[model], 
                   linewidth=2, label=model)
        
        ax.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10m')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Error (meters)", fontsize=11)
        ax.set_ylabel("Cumulative %", fontsize=11)
        ax.set_title(f"{EXPERIMENTS[exp_name]['short']}\n({EXPERIMENTS[exp_name]['test_type']})", 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 105)
    
    plt.suptitle("CDF Per Experiment (All Models)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "04_cdf_per_experiment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_cdf_per_experiment.png")


def plot_heatmap_matrix(all_results):
    """Heatmap: Models x Experiments matrix for mean error."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    # Create matrix
    matrix = np.zeros((len(MODELS), len(exp_names)))
    for i, model in enumerate(MODELS):
        for j, exp_name in enumerate(exp_names):
            if model in all_results[exp_name]:
                matrix[i, j] = all_results[exp_name][model]["mean_error"]
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels([EXPERIMENTS[e]["short"] for e in exp_names], fontsize=11)
    ax.set_yticklabels(MODELS, fontsize=11)
    
    # Add values
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}m',
                              ha='center', va='center', fontsize=12, fontweight='bold',
                              color='white' if matrix[i, j] > 30 else 'black')
    
    plt.colorbar(im, ax=ax, label='Mean Error (meters)')
    ax.set_title("Mean Error Matrix: Models × Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "05_error_heatmap_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_error_heatmap_matrix.png")


def plot_accuracy_heatmap(all_results):
    """Heatmap: Models x Experiments matrix for accuracy."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    matrix = np.zeros((len(MODELS), len(exp_names)))
    for i, model in enumerate(MODELS):
        for j, exp_name in enumerate(exp_names):
            if model in all_results[exp_name]:
                matrix[i, j] = all_results[exp_name][model]["within_10m"]
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels([EXPERIMENTS[e]["short"] for e in exp_names], fontsize=11)
    ax.set_yticklabels(MODELS, fontsize=11)
    
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                              ha='center', va='center', fontsize=12, fontweight='bold',
                              color='white' if matrix[i, j] < 30 else 'black')
    
    plt.colorbar(im, ax=ax, label='Accuracy Within 10m (%)')
    ax.set_title("Accuracy Matrix: Models × Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "06_accuracy_heatmap_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_accuracy_heatmap_matrix.png")


def plot_best_model_per_experiment(all_results):
    """Bar chart showing best model per experiment."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Best by mean error
    best_models_error = []
    best_values_error = []
    for exp_name in exp_names:
        best_model = min(
            [(m, all_results[exp_name][m]["mean_error"]) 
             for m in MODELS if m in all_results[exp_name]],
            key=lambda x: x[1]
        )
        best_models_error.append(best_model[0])
        best_values_error.append(best_model[1])
    
    colors = [MODEL_COLORS[m] for m in best_models_error]
    bars = axes[0].bar([EXPERIMENTS[e]["short"] for e in exp_names], 
                       best_values_error, color=colors, edgecolor='white')
    
    for bar, model, val in zip(bars, best_models_error, best_values_error):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{model}\n{val:.1f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[0].set_ylabel("Mean Error (meters)", fontsize=12)
    axes[0].set_title("Best Model by Mean Error", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Best by accuracy
    best_models_acc = []
    best_values_acc = []
    for exp_name in exp_names:
        best_model = max(
            [(m, all_results[exp_name][m]["within_10m"]) 
             for m in MODELS if m in all_results[exp_name]],
            key=lambda x: x[1]
        )
        best_models_acc.append(best_model[0])
        best_values_acc.append(best_model[1])
    
    colors = [MODEL_COLORS[m] for m in best_models_acc]
    bars = axes[1].bar([EXPERIMENTS[e]["short"] for e in exp_names], 
                       best_values_acc, color=colors, edgecolor='white')
    
    for bar, model, val in zip(bars, best_models_acc, best_values_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{model}\n{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[1].set_ylabel("Accuracy Within 10m (%)", fontsize=12)
    axes[1].set_title("Best Model by Accuracy", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Best Model Per Experiment", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "07_best_model_per_experiment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_best_model_per_experiment.png")


def plot_model_ranking_evolution(all_results):
    """Line chart showing how each model ranks across experiments."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Rank by mean error (lower is better)
    for model in MODELS:
        means = []
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                means.append(all_results[exp_name][model]["mean_error"])
            else:
                means.append(np.nan)
        
        axes[0].plot(range(len(exp_names)), means, color=MODEL_COLORS[model],
                    marker='o', markersize=10, linewidth=2, label=model)
    
    axes[0].set_xticks(range(len(exp_names)))
    axes[0].set_xticklabels([EXPERIMENTS[e]["short"] for e in exp_names], fontsize=10)
    axes[0].set_ylabel("Mean Error (meters)", fontsize=12)
    axes[0].set_title("Mean Error Evolution Across Experiments", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Rank by accuracy (higher is better)
    for model in MODELS:
        accs = []
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                accs.append(all_results[exp_name][model]["within_10m"])
            else:
                accs.append(np.nan)
        
        axes[1].plot(range(len(exp_names)), accs, color=MODEL_COLORS[model],
                    marker='o', markersize=10, linewidth=2, label=model)
    
    axes[1].set_xticks(range(len(exp_names)))
    axes[1].set_xticklabels([EXPERIMENTS[e]["short"] for e in exp_names], fontsize=10)
    axes[1].set_ylabel("Accuracy Within 10m (%)", fontsize=12)
    axes[1].set_title("Accuracy Evolution Across Experiments", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Model Performance Evolution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "08_model_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_model_evolution.png")


def plot_box_comparison(all_results):
    """Box plot comparing all experiments side by side."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, model in zip(axes, MODELS):
        error_data = []
        labels = []
        colors = []
        
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                error_data.append(all_results[exp_name][model]["errors"])
                labels.append(EXPERIMENTS[exp_name]["short"])
                colors.append(EXP_COLORS[exp_name])
        
        if error_data:
            bp = ax.boxplot(error_data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel("Error (meters)", fontsize=11)
        ax.set_title(f"{model}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Error Distribution Per Model Across Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "09_boxplot_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_boxplot_comparison.png")


def plot_comprehensive_summary_table(all_results):
    """Create a comprehensive summary table image."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')
    
    # Build table data
    columns = ['Experiment', 'Test Type'] + [f'{m}\nMean' for m in MODELS] + [f'{m}\n<10m' for m in MODELS]
    rows = []
    
    for exp_name in exp_names:
        row = [
            EXPERIMENTS[exp_name]["short"],
            EXPERIMENTS[exp_name]["test_type"][:15] + "..."
        ]
        
        for model in MODELS:
            if model in all_results[exp_name]:
                row.append(f"{all_results[exp_name][model]['mean_error']:.1f}m")
            else:
                row.append("-")
        
        for model in MODELS:
            if model in all_results[exp_name]:
                row.append(f"{all_results[exp_name][model]['within_10m']:.1f}%")
            else:
                row.append("-")
        
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Cross-Experiment Comparison Summary", fontsize=16, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "00_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 00_summary_table.png")


def plot_radar_chart(all_results):
    """Radar chart comparing models across experiments."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    # Metrics to compare (normalized to 0-100 scale)
    metrics = ['Within 5m', 'Within 10m', 'Within 20m', '100-Mean', '100-Median']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    for ax, model in zip(axes, MODELS):
        for exp_name in exp_names:
            if model not in all_results[exp_name]:
                continue
            
            res = all_results[exp_name][model]
            values = [
                res['within_5m'],
                res['within_10m'],
                res['within_20m'],
                max(0, 100 - res['mean_error']),  # Inverse of mean error
                max(0, 100 - res['median_error']),  # Inverse of median
            ]
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, color=EXP_COLORS[exp_name], linewidth=2, 
                   label=EXPERIMENTS[exp_name]["short"])
            ax.fill(angles, values, color=EXP_COLORS[exp_name], alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_title(f"{model}", fontsize=12, fontweight='bold', pad=20)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.suptitle("Radar Comparison: Models × Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "10_radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 10_radar_comparison.png")


def save_summary_text(all_results):
    """Save text summary."""
    exp_names = [e for e in EXPERIMENTS.keys() if e in all_results]
    
    with open(OUTPUT_DIR / "cross_experiment_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-EXPERIMENT COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENTS:\n")
        f.write("-" * 80 + "\n")
        for i, exp_name in enumerate(exp_names, 1):
            f.write(f"\n{i}. {EXPERIMENTS[exp_name]['config']}\n")
            f.write(f"   Test Type: {EXPERIMENTS[exp_name]['test_type']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESULTS BY MODEL:\n")
        f.write("=" * 80 + "\n")
        
        for model in MODELS:
            f.write(f"\n### {model} ###\n")
            f.write(f"{'Experiment':<25} {'Mean(m)':<10} {'Median(m)':<10} {'<10m':<10} {'<20m':<10}\n")
            f.write("-" * 65 + "\n")
            
            for exp_name in exp_names:
                if model in all_results[exp_name]:
                    res = all_results[exp_name][model]
                    f.write(f"{EXPERIMENTS[exp_name]['short']:<25} "
                           f"{res['mean_error']:<10.2f} "
                           f"{res['median_error']:<10.2f} "
                           f"{res['within_10m']:<10.1f} "
                           f"{res['within_20m']:<10.1f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("=" * 80 + "\n\n")
        
        # Find overall best
        best_overall = None
        best_error = float('inf')
        for exp_name in exp_names:
            for model in MODELS:
                if model in all_results[exp_name]:
                    if all_results[exp_name][model]["mean_error"] < best_error:
                        best_error = all_results[exp_name][model]["mean_error"]
                        best_overall = (model, exp_name)
        
        if best_overall:
            f.write(f"🏆 Best Overall: {best_overall[0]} in {EXPERIMENTS[best_overall[1]]['short']}\n")
            f.write(f"   Mean Error: {best_error:.2f}m\n\n")
        
        # Best per experiment
        f.write("Best Model Per Experiment:\n")
        for exp_name in exp_names:
            best_model = min(
                [(m, all_results[exp_name][m]["mean_error"]) 
                 for m in MODELS if m in all_results[exp_name]],
                key=lambda x: x[1]
            )
            f.write(f"  • {EXPERIMENTS[exp_name]['short']}: {best_model[0]} ({best_model[1]:.2f}m)\n")
        
        f.write("\n")
        f.write("Observations:\n")
        f.write("  • Augmented test (Exp3) is harder due to unseen locations\n")
        f.write("  • ConvNeXt benefits most from lower LR (0.0001)\n")
        f.write("  • EfficientNet was best with full data and higher LR\n")
    
    print(f"  ✓ cross_experiment_summary.txt")


def main():
    """Generate all cross-experiment comparison plots."""
    print("=" * 70)
    print("CROSS-EXPERIMENT COMPARISON")
    print("=" * 70)
    
    setup_dirs()
    
    print("\nLoading results from all experiments...")
    all_results = load_all_results()
    
    exp_count = len([e for e in EXPERIMENTS.keys() if e in all_results])
    print(f"Loaded {exp_count} experiments")
    
    for exp_name in EXPERIMENTS.keys():
        if exp_name in all_results:
            models_loaded = [m for m in MODELS if m in all_results[exp_name]]
            print(f"  • {EXPERIMENTS[exp_name]['short']}: {len(models_loaded)} models")
    
    print("\n📊 Generating comparison plots...")
    
    plot_comprehensive_summary_table(all_results)
    plot_mean_error_grouped(all_results)
    plot_accuracy_grouped(all_results)
    plot_cdf_per_model(all_results)
    plot_cdf_per_experiment(all_results)
    plot_heatmap_matrix(all_results)
    plot_accuracy_heatmap(all_results)
    plot_best_model_per_experiment(all_results)
    plot_model_ranking_evolution(all_results)
    plot_box_comparison(all_results)
    plot_radar_chart(all_results)
    
    print("\n📝 Saving summary...")
    save_summary_text(all_results)
    
    print("\n" + "=" * 70)
    print("✅ CROSS-EXPERIMENT COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Plots: {OUTPUT_DIR}/plots/")
    
    # Count plots
    n_plots = len(list((OUTPUT_DIR / "plots").glob("*.png")))
    print(f"Total plots generated: {n_plots}")


if __name__ == "__main__":
    main()
