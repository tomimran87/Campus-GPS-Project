#!/usr/bin/env python3
"""
Comprehensive Comparison: All Experiments
==========================================

Creates comparison plots for:
1. All 100 epoch experiments
2. All LR=0.0001 experiments  
3. All experiments overall
4. Geographic analysis for all experiments

Output: testing/comprehensive_comparison/
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

# ALL experiments organized by properties
ALL_EXPERIMENTS = {
    # 30 epoch experiments
    "30ep_Full_lr0001": {
        "path": Path("testing/30epochs"),
        "epochs": 30, "lr": 0.0001, "data": "full", "test_type": "regular",
        "display": "30ep Full"
    },
    "30ep_Half_lr0001": {
        "path": Path("testing/30epochs_lr0.0001_halfdata"),
        "epochs": 30, "lr": 0.0001, "data": "half", "test_type": "regular",
        "display": "30ep Half"
    },
    "30ep_Half_lr0001_Aug": {
        "path": Path("testing/30ep_lr0001_halfdata_augtest"),
        "epochs": 30, "lr": 0.0001, "data": "half", "test_type": "augmented",
        "display": "30ep Half+Aug"
    },
    # 50 epoch experiments
    "50ep_Half_lr0001": {
        "path": Path("testing/50ep_lr0001_halfdata"),
        "epochs": 50, "lr": 0.0001, "data": "half", "test_type": "regular",
        "display": "50ep Half"
    },
    "50ep_Half_lr0001_Aug": {
        "path": Path("testing/50ep_lr0001_halfdata_augtest"),
        "epochs": 50, "lr": 0.0001, "data": "half", "test_type": "augmented",
        "display": "50ep Half+Aug"
    },
    "50ep_Full_lr001": {
        "path": Path("testing/50ep_lr001_fulldata"),
        "epochs": 50, "lr": 0.001, "data": "full", "test_type": "regular",
        "display": "50ep Full (lr=0.001)"
    },
    # 100 epoch experiments
    "100ep_Half_lr0001": {
        "path": Path("testing/100ep_lr0001_halfdata"),
        "epochs": 100, "lr": 0.0001, "data": "half", "test_type": "regular",
        "display": "100ep Half"
    },
    "100ep_Half_lr0001_Aug": {
        "path": Path("testing/100ep_lr0001_halfdata_augtest"),
        "epochs": 100, "lr": 0.0001, "data": "half", "test_type": "augmented",
        "display": "100ep Half+Aug"
    },
    "100ep_Full_lr001": {
        "path": Path("testing/100ep_lr001_fulldata"),
        "epochs": 100, "lr": 0.001, "data": "full", "test_type": "regular",
        "display": "100ep Full (lr=0.001)"
    },
    "100ep_Full_lr0001": {
        "path": Path("testing/100ep_lr0001_fulldata"),
        "epochs": 100, "lr": 0.0001, "data": "full", "test_type": "regular",
        "display": "100ep Full (lr=0.0001)"
    },
}

OUTPUT_DIR = Path("testing/comprehensive_comparison")


def load_experiment_results(exp_path):
    """Load results for all models in an experiment."""
    results = {}
    
    for model in MODELS:
        model_dir = exp_path / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
            continue
        
        errors = np.load(errors_path)
        
        # Load predictions and labels if available
        preds_path = model_dir / f"{model.lower()}_predictions.npy"
        labels_path = model_dir / f"{model.lower()}_labels.npy"
        
        preds = np.load(preds_path) if preds_path.exists() else None
        labels = np.load(labels_path) if labels_path.exists() else None
        
        results_file = model_dir / "test_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                test_results = json.load(f)
        else:
            test_results = {}
        
        history_file = model_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {}
        
        results[model] = {
            "errors": errors,
            "predictions": preds,
            "labels": labels,
            "history": history,
            **test_results
        }
    
    return results


def load_all_experiments():
    """Load all available experiments."""
    all_results = {}
    
    print("\nLoading experiments...")
    for exp_name, config in ALL_EXPERIMENTS.items():
        if config["path"].exists():
            results = load_experiment_results(config["path"])
            if results:
                all_results[exp_name] = {
                    "results": results,
                    "config": config
                }
                print(f"  ✓ {config['display']}: {len(results)} models")
            else:
                print(f"  ⚠ {config['display']}: No data found")
        else:
            print(f"  ✗ {config['display']}: Path not found")
    
    return all_results


# ============================================
# 100 EPOCHS COMPARISON
# ============================================
def plot_100ep_comparison(all_results):
    """Compare all 100 epoch experiments."""
    print("\n📊 Generating 100 Epochs Comparison...")
    
    output_dir = OUTPUT_DIR / "100_epochs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_100 = {k: v for k, v in all_results.items() if v["config"]["epochs"] == 100}
    
    if len(exp_100) < 2:
        print("  ⚠ Not enough 100 epoch experiments")
        return
    
    exp_names = list(exp_100.keys())
    display_names = [ALL_EXPERIMENTS[e]["display"] for e in exp_names]
    
    # 1. Bar chart - Mean Error
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    ax = axes[0]
    for i, model in enumerate(MODELS):
        means = []
        for exp in exp_names:
            if model in exp_100[exp]["results"]:
                means.append(exp_100[exp]["results"][model].get('mean_error', 0))
            else:
                means.append(0)
        ax.bar(x + i*width, means, width, label=model, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Mean Error - 100 Epoch Experiments')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Within 10m
    ax = axes[1]
    for i, model in enumerate(MODELS):
        acc = []
        for exp in exp_names:
            if model in exp_100[exp]["results"]:
                acc.append(exp_100[exp]["results"][model].get('within_10m', 0))
            else:
                acc.append(0)
        ax.bar(x + i*width, acc, width, label=model, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Within 10m Accuracy - 100 Epoch Experiments')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_100ep_bar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_100ep_bar_comparison.png")
    
    # 2. Box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        data = []
        labels = []
        
        for exp in exp_names:
            if model in exp_100[exp]["results"]:
                data.append(exp_100[exp]["results"][model]['errors'])
                labels.append(ALL_EXPERIMENTS[exp]["display"].replace(" ", "\n"))
        
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(0.6)
        
        ax.set_ylabel('Error (m)')
        ax.set_title(f'{model}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Error Distribution - 100 Epoch Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "02_100ep_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_100ep_boxplots.png")
    
    # 3. CDF comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    linestyles = ['-', '--', ':', '-.']
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        
        for j, exp in enumerate(exp_names):
            if model in exp_100[exp]["results"]:
                errors = np.sort(exp_100[exp]["results"][model]['errors'])
                cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
                ax.plot(errors, cdf, linestyle=linestyles[j % len(linestyles)], 
                       linewidth=2, label=ALL_EXPERIMENTS[exp]["display"])
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=10, color='black', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Error (meters)')
        ax.set_ylabel('Cumulative %')
        ax.set_title(f'{model}')
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('CDF of Errors - 100 Epoch Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "03_100ep_cdf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_100ep_cdf.png")
    
    # 4. Summary table
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    columns = ['Experiment', 'Model', 'Mean', 'Median', '<5m', '<10m', '<20m', 'P90']
    rows = []
    
    for exp in exp_names:
        for model in MODELS:
            if model in exp_100[exp]["results"]:
                r = exp_100[exp]["results"][model]
                rows.append([
                    ALL_EXPERIMENTS[exp]["display"],
                    model,
                    f"{r.get('mean_error', 0):.2f}m",
                    f"{r.get('median_error', 0):.2f}m",
                    f"{r.get('within_5m', 0):.1f}%",
                    f"{r.get('within_10m', 0):.1f}%",
                    f"{r.get('within_20m', 0):.1f}%",
                    f"{r.get('p90_error', 0):.1f}m",
                ])
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, row in enumerate(rows):
        model = row[1]
        table[(i+1, 1)].set_facecolor(MODEL_COLORS.get(model, 'white'))
        table[(i+1, 1)].set_text_props(color='white', fontweight='bold')
    
    plt.title("100 Epoch Experiments - Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "04_100ep_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_100ep_summary_table.png")


# ============================================
# LR=0.0001 COMPARISON
# ============================================
def plot_lr0001_comparison(all_results):
    """Compare all LR=0.0001 experiments."""
    print("\n📊 Generating LR=0.0001 Comparison...")
    
    output_dir = OUTPUT_DIR / "lr_0001"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_lr0001 = {k: v for k, v in all_results.items() if v["config"]["lr"] == 0.0001}
    
    if len(exp_lr0001) < 2:
        print("  ⚠ Not enough LR=0.0001 experiments")
        return
    
    exp_names = list(exp_lr0001.keys())
    display_names = [ALL_EXPERIMENTS[e]["display"] for e in exp_names]
    
    # 1. Heatmap of results
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Mean error heatmap
    mean_data = np.zeros((len(MODELS), len(exp_names)))
    for j, exp in enumerate(exp_names):
        for i, model in enumerate(MODELS):
            if model in exp_lr0001[exp]["results"]:
                mean_data[i, j] = exp_lr0001[exp]["results"][model].get('mean_error', np.nan)
            else:
                mean_data[i, j] = np.nan
    
    ax = axes[0]
    im = ax.imshow(mean_data, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title('Mean Error (m) - Lower is Better', fontweight='bold')
    
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(mean_data[i, j]):
                color = 'white' if mean_data[i, j] > 15 else 'black'
                ax.text(j, i, f'{mean_data[i, j]:.1f}', ha='center', va='center', 
                       fontsize=9, color=color)
    
    plt.colorbar(im, ax=ax, label='Mean Error (m)')
    
    # Within 10m heatmap
    acc_data = np.zeros((len(MODELS), len(exp_names)))
    for j, exp in enumerate(exp_names):
        for i, model in enumerate(MODELS):
            if model in exp_lr0001[exp]["results"]:
                acc_data[i, j] = exp_lr0001[exp]["results"][model].get('within_10m', np.nan)
            else:
                acc_data[i, j] = np.nan
    
    ax = axes[1]
    im = ax.imshow(acc_data, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title('Within 10m (%) - Higher is Better', fontweight='bold')
    
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(acc_data[i, j]):
                color = 'white' if acc_data[i, j] < 50 else 'black'
                ax.text(j, i, f'{acc_data[i, j]:.1f}', ha='center', va='center', 
                       fontsize=9, color=color)
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.suptitle('All LR=0.0001 Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "01_lr0001_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_lr0001_heatmap.png")
    
    # 2. Epoch progression for LR=0.0001
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group by data type
    half_regular = ["30ep_Half_lr0001", "50ep_Half_lr0001", "100ep_Half_lr0001"]
    half_aug = ["30ep_Half_lr0001_Aug", "50ep_Half_lr0001_Aug", "100ep_Half_lr0001_Aug"]
    full_exps = ["30ep_Full_lr0001", "100ep_Full_lr0001"]
    
    # Half data regular - Mean error
    ax = axes[0, 0]
    for model in MODELS:
        epochs = []
        means = []
        for exp in half_regular:
            if exp in exp_lr0001 and model in exp_lr0001[exp]["results"]:
                epochs.append(ALL_EXPERIMENTS[exp]["epochs"])
                means.append(exp_lr0001[exp]["results"][model].get('mean_error', np.nan))
        if epochs:
            ax.plot(epochs, means, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Half Data (Regular Test)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Half data augmented - Mean error
    ax = axes[0, 1]
    for model in MODELS:
        epochs = []
        means = []
        for exp in half_aug:
            if exp in exp_lr0001 and model in exp_lr0001[exp]["results"]:
                epochs.append(ALL_EXPERIMENTS[exp]["epochs"])
                means.append(exp_lr0001[exp]["results"][model].get('mean_error', np.nan))
        if epochs:
            ax.plot(epochs, means, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Half Data (Augmented Test)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Full data - Mean error
    ax = axes[1, 0]
    for model in MODELS:
        epochs = []
        means = []
        for exp in full_exps:
            if exp in exp_lr0001 and model in exp_lr0001[exp]["results"]:
                epochs.append(ALL_EXPERIMENTS[exp]["epochs"])
                means.append(exp_lr0001[exp]["results"][model].get('mean_error', np.nan))
        if epochs:
            ax.plot(epochs, means, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Full Data')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Best model per experiment
    ax = axes[1, 1]
    best_per_exp = []
    for exp in exp_names:
        best_model = None
        best_error = float('inf')
        for model in MODELS:
            if model in exp_lr0001[exp]["results"]:
                err = exp_lr0001[exp]["results"][model].get('mean_error', float('inf'))
                if err < best_error:
                    best_error = err
                    best_model = model
        best_per_exp.append((ALL_EXPERIMENTS[exp]["display"], best_model, best_error))
    
    colors = [MODEL_COLORS[b[1]] if b[1] else 'gray' for b in best_per_exp]
    bars = ax.barh(range(len(best_per_exp)), [b[2] for b in best_per_exp], color=colors)
    ax.set_yticks(range(len(best_per_exp)))
    ax.set_yticklabels([f"{b[0]}\n({b[1]})" for b in best_per_exp])
    ax.set_xlabel('Mean Error (m)')
    ax.set_title('Best Model per Experiment')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, bp) in enumerate(zip(bars, best_per_exp)):
        ax.text(bp[2] + 0.3, i, f'{bp[2]:.1f}m', va='center', fontsize=9)
    
    plt.suptitle('LR=0.0001 Experiments - Epoch Progression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "02_lr0001_epoch_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_lr0001_epoch_progression.png")


# ============================================
# OVERALL COMPARISON
# ============================================
def plot_overall_comparison(all_results):
    """Create overall comparison of all experiments."""
    print("\n📊 Generating Overall Comparison...")
    
    output_dir = OUTPUT_DIR / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_names = list(all_results.keys())
    
    # 1. Master heatmap
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    display_names = [ALL_EXPERIMENTS[e]["display"] for e in exp_names]
    
    # Mean error
    mean_data = np.zeros((len(MODELS), len(exp_names)))
    for j, exp in enumerate(exp_names):
        for i, model in enumerate(MODELS):
            if model in all_results[exp]["results"]:
                mean_data[i, j] = all_results[exp]["results"][model].get('mean_error', np.nan)
            else:
                mean_data[i, j] = np.nan
    
    ax = axes[0]
    im = ax.imshow(mean_data, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title('Mean Error (m)', fontweight='bold')
    
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(mean_data[i, j]):
                color = 'white' if mean_data[i, j] > 20 else 'black'
                ax.text(j, i, f'{mean_data[i, j]:.1f}', ha='center', va='center', 
                       fontsize=7, color=color)
    
    plt.colorbar(im, ax=ax)
    
    # Within 10m
    acc_data = np.zeros((len(MODELS), len(exp_names)))
    for j, exp in enumerate(exp_names):
        for i, model in enumerate(MODELS):
            if model in all_results[exp]["results"]:
                acc_data[i, j] = all_results[exp]["results"][model].get('within_10m', np.nan)
            else:
                acc_data[i, j] = np.nan
    
    ax = axes[1]
    im = ax.imshow(acc_data, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title('Within 10m (%)', fontweight='bold')
    
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(acc_data[i, j]):
                color = 'white' if acc_data[i, j] < 40 else 'black'
                ax.text(j, i, f'{acc_data[i, j]:.0f}', ha='center', va='center', 
                       fontsize=7, color=color)
    
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('All Experiments Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "01_master_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_master_heatmap.png")
    
    # 2. Top 15 rankings
    rankings = []
    for exp in all_results:
        for model in MODELS:
            if model in all_results[exp]["results"]:
                rankings.append({
                    'experiment': ALL_EXPERIMENTS[exp]["display"],
                    'model': model,
                    'mean_error': all_results[exp]["results"][model].get('mean_error', float('inf')),
                    'within_10m': all_results[exp]["results"][model].get('within_10m', 0),
                    'within_5m': all_results[exp]["results"][model].get('within_5m', 0),
                })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # By mean error
    rankings.sort(key=lambda x: x['mean_error'])
    top15 = rankings[:15]
    
    ax = axes[0]
    labels = [f"{r['model']}\n{r['experiment']}" for r in top15]
    values = [r['mean_error'] for r in top15]
    colors = [MODEL_COLORS[r['model']] for r in top15]
    
    bars = ax.barh(range(len(top15)), values, color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Error (m)')
    ax.set_title('Top 15 - Lowest Mean Error', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.2, i, f'{val:.2f}m', va='center', fontsize=8)
    
    # By within 10m
    rankings.sort(key=lambda x: x['within_10m'], reverse=True)
    top15 = rankings[:15]
    
    ax = axes[1]
    labels = [f"{r['model']}\n{r['experiment']}" for r in top15]
    values = [r['within_10m'] for r in top15]
    colors = [MODEL_COLORS[r['model']] for r in top15]
    
    bars = ax.barh(range(len(top15)), values, color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Within 10m (%)')
    ax.set_title('Top 15 - Highest <10m Accuracy', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_top15_rankings.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_top15_rankings.png")
    
    # 3. Model wins by category
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    categories = [
        ("100 Epochs", lambda e: ALL_EXPERIMENTS[e]["epochs"] == 100),
        ("LR=0.0001", lambda e: ALL_EXPERIMENTS[e]["lr"] == 0.0001),
        ("Full Data", lambda e: ALL_EXPERIMENTS[e]["data"] == "full"),
        ("Half Data", lambda e: ALL_EXPERIMENTS[e]["data"] == "half"),
    ]
    
    for idx, (cat_name, filter_fn) in enumerate(categories):
        ax = axes[idx // 2, idx % 2]
        
        cat_exps = [e for e in exp_names if filter_fn(e)]
        
        wins = {m: 0 for m in MODELS}
        for exp in cat_exps:
            best_model = None
            best_error = float('inf')
            for model in MODELS:
                if model in all_results[exp]["results"]:
                    err = all_results[exp]["results"][model].get('mean_error', float('inf'))
                    if err < best_error:
                        best_error = err
                        best_model = model
            if best_model:
                wins[best_model] += 1
        
        colors = [MODEL_COLORS[m] for m in MODELS]
        bars = ax.bar(MODELS, [wins[m] for m in MODELS], color=colors)
        ax.set_ylabel('Number of Wins')
        ax.set_title(f'{cat_name} ({len(cat_exps)} experiments)')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, m in zip(bars, MODELS):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(wins[m]), ha='center', fontweight='bold')
    
    plt.suptitle('Model Wins by Category (Lowest Mean Error)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "03_model_wins.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_model_wins.png")
    
    # 4. Comprehensive table
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('off')
    
    columns = ['Experiment', 'Epochs', 'LR', 'Data', 'Test', 'Model', 'Mean', 'Median', '<10m', '<20m']
    rows = []
    
    for exp in exp_names:
        cfg = ALL_EXPERIMENTS[exp]
        for model in MODELS:
            if model in all_results[exp]["results"]:
                r = all_results[exp]["results"][model]
                rows.append([
                    cfg["display"],
                    str(cfg["epochs"]),
                    str(cfg["lr"]),
                    cfg["data"],
                    cfg["test_type"],
                    model,
                    f"{r.get('mean_error', 0):.2f}",
                    f"{r.get('median_error', 0):.2f}",
                    f"{r.get('within_10m', 0):.1f}%",
                    f"{r.get('within_20m', 0):.1f}%",
                ])
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.1, 1.4)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Complete Experiments Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "04_complete_table.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_complete_table.png")
    
    # Save JSON summary
    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": len(exp_names),
        "rankings": rankings[:20],
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("  ✓ summary.json")


def main():
    """Generate all comparison plots."""
    print("=" * 70)
    print("COMPREHENSIVE COMPARISON - ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    all_results = load_all_experiments()
    
    if not all_results:
        print("ERROR: No experiments found!")
        return
    
    # Generate comparison plots
    plot_100ep_comparison(all_results)
    plot_lr0001_comparison(all_results)
    plot_overall_comparison(all_results)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
