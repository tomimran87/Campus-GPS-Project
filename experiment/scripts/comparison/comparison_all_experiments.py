#!/usr/bin/env python3
"""
Comprehensive All Experiments Comparison
=========================================

Compares ALL experiments including:
1. All 100 epoch experiments (halfdata, fulldata, augtest)
2. All experiments across all configurations

Output: testing/comparison_all_experiments/
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
MODEL_MARKERS = {"ResNet18": "o", "EfficientNet": "s", "ConvNeXt": "^"}

# ALL experiments organized by category
ALL_EXPERIMENTS = {
    # 30 epoch experiments
    "30ep Full": {
        "path": Path("testing/30epochs"),
        "epochs": 30, "lr": 0.0001, "data": "full", "test_type": "regular"
    },
    "30ep Half": {
        "path": Path("testing/30epochs_lr0.0001_halfdata"),
        "epochs": 30, "lr": 0.0001, "data": "half", "test_type": "regular"
    },
    "30ep Half+Aug": {
        "path": Path("testing/30ep_lr0001_halfdata_augtest"),
        "epochs": 30, "lr": 0.0001, "data": "half", "test_type": "augmented"
    },
    # 50 epoch experiments
    "50ep Half": {
        "path": Path("testing/50ep_lr0001_halfdata"),
        "epochs": 50, "lr": 0.0001, "data": "half", "test_type": "regular"
    },
    "50ep Half+Aug": {
        "path": Path("testing/50ep_lr0001_halfdata_augtest"),
        "epochs": 50, "lr": 0.0001, "data": "half", "test_type": "augmented"
    },
    "50ep Full (lr=0.001)": {
        "path": Path("testing/50ep_lr001_fulldata"),
        "epochs": 50, "lr": 0.001, "data": "full", "test_type": "regular"
    },
    # 100 epoch experiments
    "100ep Half": {
        "path": Path("testing/100ep_lr0001_halfdata"),
        "epochs": 100, "lr": 0.0001, "data": "half", "test_type": "regular"
    },
    "100ep Half+Aug": {
        "path": Path("testing/100ep_lr0001_halfdata_augtest"),
        "epochs": 100, "lr": 0.0001, "data": "half", "test_type": "augmented"
    },
    "100ep Full (lr=0.001)": {
        "path": Path("testing/100ep_lr001_fulldata"),
        "epochs": 100, "lr": 0.001, "data": "full", "test_type": "regular"
    },
}

# Color scheme for experiments
EXP_COLORS = {
    "30ep Full": "#1f77b4",
    "30ep Half": "#aec7e8",
    "30ep Half+Aug": "#ff7f0e",
    "50ep Half": "#2ca02c",
    "50ep Half+Aug": "#98df8a",
    "50ep Full (lr=0.001)": "#d62728",
    "100ep Half": "#9467bd",
    "100ep Half+Aug": "#c5b0d5",
    "100ep Full (lr=0.001)": "#8c564b",
}

OUTPUT_DIR = Path("testing/comparison_all_experiments")


def load_experiment_results(exp_path):
    """Load results for all models in an experiment."""
    results = {}
    
    for model in MODELS:
        model_dir = exp_path / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
            continue
        
        errors = np.load(errors_path)
        
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
                all_results[exp_name] = results
                print(f"  ✓ {exp_name}: {len(results)} models")
            else:
                print(f"  ⚠ {exp_name}: No data found")
        else:
            print(f"  ✗ {exp_name}: Path not found")
    
    return all_results


def plot_100ep_comparison(all_results):
    """Compare all 100 epoch experiments."""
    print("\n📊 Generating 100 Epochs Comparison...")
    
    exp_100 = [e for e in all_results.keys() if "100ep" in e]
    
    if len(exp_100) < 2:
        print("  ⚠ Not enough 100 epoch experiments to compare")
        return
    
    # 1. Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(exp_100))
    width = 0.25
    
    # Mean error
    ax = axes[0]
    for i, model in enumerate(MODELS):
        means = []
        for exp in exp_100:
            if model in all_results[exp]:
                means.append(all_results[exp][model].get('mean_error', 0))
            else:
                means.append(0)
        ax.bar(x + i*width, means, width, label=model, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Mean Error - 100 Epoch Experiments')
    ax.set_xticks(x + width)
    ax.set_xticklabels([e.replace("100ep ", "") for e in exp_100], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Within 10m
    ax = axes[1]
    for i, model in enumerate(MODELS):
        acc = []
        for exp in exp_100:
            if model in all_results[exp]:
                acc.append(all_results[exp][model].get('within_10m', 0))
            else:
                acc.append(0)
        ax.bar(x + i*width, acc, width, label=model, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Within 10m Accuracy - 100 Epoch Experiments')
    ax.set_xticks(x + width)
    ax.set_xticklabels([e.replace("100ep ", "") for e in exp_100], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_100ep_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_100ep_comparison.png")
    
    # 2. Box plots for 100ep experiments
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        data = []
        labels = []
        
        for exp in exp_100:
            if model in all_results[exp]:
                data.append(all_results[exp][model]['errors'])
                labels.append(exp.replace("100ep ", ""))
        
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(0.6)
        
        ax.set_ylabel('Error (m)')
        ax.set_title(f'{model}')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('Error Distribution - 100 Epoch Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_100ep_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_100ep_boxplots.png")


def plot_all_experiments_heatmap(all_results):
    """Create heatmap of all experiments."""
    print("\n📊 Generating All Experiments Heatmap...")
    
    exp_names = list(all_results.keys())
    
    # Mean error heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare data for mean error
    mean_data = np.zeros((len(MODELS), len(exp_names)))
    for j, exp in enumerate(exp_names):
        for i, model in enumerate(MODELS):
            if model in all_results[exp]:
                mean_data[i, j] = all_results[exp][model].get('mean_error', np.nan)
            else:
                mean_data[i, j] = np.nan
    
    ax = axes[0]
    im = ax.imshow(mean_data, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title('Mean Error (m) - Lower is Better', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(mean_data[i, j]):
                ax.text(j, i, f'{mean_data[i, j]:.1f}', ha='center', va='center', 
                       fontsize=9, color='white' if mean_data[i, j] > 20 else 'black')
    
    plt.colorbar(im, ax=ax, label='Mean Error (m)')
    
    # Prepare data for within 10m
    acc_data = np.zeros((len(MODELS), len(exp_names)))
    for j, exp in enumerate(exp_names):
        for i, model in enumerate(MODELS):
            if model in all_results[exp]:
                acc_data[i, j] = all_results[exp][model].get('within_10m', np.nan)
            else:
                acc_data[i, j] = np.nan
    
    ax = axes[1]
    im = ax.imshow(acc_data, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title('Within 10m Accuracy (%) - Higher is Better', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(MODELS)):
        for j in range(len(exp_names)):
            if not np.isnan(acc_data[i, j]):
                ax.text(j, i, f'{acc_data[i, j]:.1f}', ha='center', va='center', 
                       fontsize=9, color='white' if acc_data[i, j] < 40 else 'black')
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.suptitle('All Experiments Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_all_experiments_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_all_experiments_heatmap.png")


def plot_model_comparison_across_all(all_results):
    """Compare each model across all experiments."""
    print("\n📊 Generating Model Comparison Across All Experiments...")
    
    exp_names = list(all_results.keys())
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        
        # Get data for this model
        means = []
        medians = []
        within_10m = []
        valid_exps = []
        
        for exp in exp_names:
            if model in all_results[exp]:
                means.append(all_results[exp][model].get('mean_error', 0))
                medians.append(all_results[exp][model].get('median_error', 0))
                within_10m.append(all_results[exp][model].get('within_10m', 0))
                valid_exps.append(exp)
        
        x = np.arange(len(valid_exps))
        width = 0.35
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width/2, means, width, label='Mean Error', 
                       color=MODEL_COLORS[model], alpha=0.7)
        bars2 = ax.bar(x + width/2, medians, width, label='Median Error', 
                       color=MODEL_COLORS[model], alpha=0.4)
        
        line = ax2.plot(x, within_10m, 'ko-', linewidth=2, markersize=8, label='Within 10m %')
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Error (m)', color=MODEL_COLORS[model])
        ax2.set_ylabel('Within 10m (%)', color='black')
        ax.set_title(f'{model} Performance Across All Experiments', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_exps, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_model_comparison_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_model_comparison_all.png")


def plot_epoch_effect(all_results):
    """Show effect of epochs on performance."""
    print("\n📊 Generating Epoch Effect Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Half data experiments (regular test)
    ax = axes[0, 0]
    half_exps = ["30ep Half", "50ep Half", "100ep Half"]
    epochs = [30, 50, 100]
    
    for model in MODELS:
        means = []
        valid_epochs = []
        for exp, ep in zip(half_exps, epochs):
            if exp in all_results and model in all_results[exp]:
                means.append(all_results[exp][model].get('mean_error', np.nan))
                valid_epochs.append(ep)
        if means:
            ax.plot(valid_epochs, means, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Half Data (Regular Test)')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Half data experiments (augmented test)
    ax = axes[0, 1]
    aug_exps = ["30ep Half+Aug", "50ep Half+Aug", "100ep Half+Aug"]
    
    for model in MODELS:
        means = []
        valid_epochs = []
        for exp, ep in zip(aug_exps, epochs):
            if exp in all_results and model in all_results[exp]:
                means.append(all_results[exp][model].get('mean_error', np.nan))
                valid_epochs.append(ep)
        if means:
            ax.plot(valid_epochs, means, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Half Data (Augmented Test)')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Within 10m - Half data regular
    ax = axes[1, 0]
    for model in MODELS:
        acc = []
        valid_epochs = []
        for exp, ep in zip(half_exps, epochs):
            if exp in all_results and model in all_results[exp]:
                acc.append(all_results[exp][model].get('within_10m', np.nan))
                valid_epochs.append(ep)
        if acc:
            ax.plot(valid_epochs, acc, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Within 10m (%)')
    ax.set_title('Half Data (Regular Test) - Accuracy')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Within 10m - Augmented test
    ax = axes[1, 1]
    for model in MODELS:
        acc = []
        valid_epochs = []
        for exp, ep in zip(aug_exps, epochs):
            if exp in all_results and model in all_results[exp]:
                acc.append(all_results[exp][model].get('within_10m', np.nan))
                valid_epochs.append(ep)
        if acc:
            ax.plot(valid_epochs, acc, '-o', color=MODEL_COLORS[model], 
                   linewidth=2, markersize=10, label=model)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Within 10m (%)')
    ax.set_title('Half Data (Augmented Test) - Accuracy')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Effect of Training Epochs on Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_epoch_effect.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_epoch_effect.png")


def plot_data_and_test_type_effect(all_results):
    """Compare data size and test type effects."""
    print("\n📊 Generating Data/Test Type Effect Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compare 100ep experiments: Half vs Half+Aug
    ax = axes[0]
    
    comparisons = [
        ("100ep Half", "100ep Half+Aug"),
        ("50ep Half", "50ep Half+Aug"),
        ("30ep Half", "30ep Half+Aug"),
    ]
    
    x = np.arange(len(comparisons))
    width = 0.25
    
    for i, model in enumerate(MODELS):
        regular = []
        augmented = []
        for reg, aug in comparisons:
            if reg in all_results and model in all_results[reg]:
                regular.append(all_results[reg][model].get('mean_error', 0))
            else:
                regular.append(0)
            if aug in all_results and model in all_results[aug]:
                augmented.append(all_results[aug][model].get('mean_error', 0))
            else:
                augmented.append(0)
        
        ax.bar(x + i*width - width, regular, width/2, 
               label=f'{model} Regular' if i == 0 else "", 
               color=MODEL_COLORS[model], alpha=0.6)
        ax.bar(x + i*width - width/2, augmented, width/2, 
               label=f'{model} Aug' if i == 0 else "", 
               color=MODEL_COLORS[model], alpha=1.0, hatch='///')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Regular Test vs Augmented Test')
    ax.set_xticks(x)
    ax.set_xticklabels(['100ep', '50ep', '30ep'])
    ax.legend(['Regular', 'Augmented'], loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # ConvNeXt across all experiments
    ax = axes[1]
    model = "ConvNeXt"
    
    exp_names = list(all_results.keys())
    means = []
    valid_exps = []
    colors = []
    
    for exp in exp_names:
        if model in all_results[exp]:
            means.append(all_results[exp][model].get('mean_error', 0))
            valid_exps.append(exp)
            colors.append(EXP_COLORS.get(exp, 'gray'))
    
    bars = ax.barh(range(len(valid_exps)), means, color=colors)
    ax.set_yticks(range(len(valid_exps)))
    ax.set_yticklabels(valid_exps)
    ax.set_xlabel('Mean Error (m)')
    ax.set_title(f'{model} Performance Across All Experiments')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(val + 0.5, i, f'{val:.1f}m', va='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_data_test_type_effect.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_data_test_type_effect.png")


def create_comprehensive_summary_table(all_results):
    """Create comprehensive summary table."""
    print("\n📊 Generating Comprehensive Summary Table...")
    
    exp_names = list(all_results.keys())
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')
    
    # Build table data
    columns = ['Experiment', 'Model', 'Mean', 'Median', 'Std', '<5m', '<10m', '<20m', '<50m', 'P90']
    rows = []
    
    for exp in exp_names:
        for model in MODELS:
            if model in all_results[exp]:
                r = all_results[exp][model]
                row = [
                    exp,
                    model,
                    f"{r.get('mean_error', 0):.1f}",
                    f"{r.get('median_error', 0):.1f}",
                    f"{r.get('std_error', 0):.1f}",
                    f"{r.get('within_5m', 0):.0f}%",
                    f"{r.get('within_10m', 0):.0f}%",
                    f"{r.get('within_20m', 0):.0f}%",
                    f"{r.get('within_50m', 0):.0f}%",
                    f"{r.get('p90_error', 0):.1f}",
                ]
                rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.5)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Model column styling
    for i, row in enumerate(rows):
        model = row[1]
        table[(i+1, 1)].set_facecolor(MODEL_COLORS.get(model, 'white'))
        table[(i+1, 1)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Comprehensive Results Summary - All Experiments", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_comprehensive_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_comprehensive_table.png")


def plot_best_models_ranking(all_results):
    """Rank all model-experiment combinations."""
    print("\n📊 Generating Best Models Ranking...")
    
    # Collect all results
    rankings = []
    for exp in all_results:
        for model in MODELS:
            if model in all_results[exp]:
                rankings.append({
                    'experiment': exp,
                    'model': model,
                    'mean_error': all_results[exp][model].get('mean_error', float('inf')),
                    'within_10m': all_results[exp][model].get('within_10m', 0),
                })
    
    # Sort by mean error
    rankings.sort(key=lambda x: x['mean_error'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 10 by mean error
    ax = axes[0]
    top10 = rankings[:min(10, len(rankings))]
    labels = [f"{r['model']}\n{r['experiment']}" for r in top10]
    values = [r['mean_error'] for r in top10]
    colors = [MODEL_COLORS[r['model']] for r in top10]
    
    bars = ax.barh(range(len(top10)), values, color=colors)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean Error (m)')
    ax.set_title('Top 10 Best Results (Lowest Mean Error)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.3, i, f'{val:.1f}m', va='center', fontsize=9)
    
    # Sort by within 10m
    rankings.sort(key=lambda x: x['within_10m'], reverse=True)
    
    ax = axes[1]
    top10 = rankings[:min(10, len(rankings))]
    labels = [f"{r['model']}\n{r['experiment']}" for r in top10]
    values = [r['within_10m'] for r in top10]
    colors = [MODEL_COLORS[r['model']] for r in top10]
    
    bars = ax.barh(range(len(top10)), values, color=colors)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Within 10m (%)')
    ax.set_title('Top 10 Best Results (Highest <10m Accuracy)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_best_models_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_best_models_ranking.png")


def plot_cdf_all_100ep(all_results):
    """CDF comparison for all 100 epoch experiments."""
    print("\n📊 Generating CDF for 100 Epoch Experiments...")
    
    exp_100 = [e for e in all_results.keys() if "100ep" in e]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    linestyles = ['-', '--', ':']
    
    for i, model in enumerate(MODELS):
        ax = axes[i]
        
        for j, exp in enumerate(exp_100):
            if model in all_results[exp]:
                errors = np.sort(all_results[exp][model]['errors'])
                cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
                ax.plot(errors, cdf, linestyle=linestyles[j % len(linestyles)], 
                       linewidth=2, label=exp.replace("100ep ", ""))
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=10, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=20, color='black', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Error (meters)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title(f'{model}')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    
    plt.suptitle('CDF of Errors - 100 Epoch Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_cdf_100ep.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_cdf_100ep.png")


def save_results_json(all_results):
    """Save all results to JSON."""
    print("\n📊 Saving results to JSON...")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for exp in all_results:
        json_results[exp] = {}
        for model in all_results[exp]:
            json_results[exp][model] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in all_results[exp][model].items()
                if k != 'history'  # Skip history to reduce size
            }
    
    with open(OUTPUT_DIR / "all_experiments_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("  ✓ all_experiments_results.json")


def main():
    """Generate all comparison plots."""
    print("=" * 70)
    print("COMPREHENSIVE ALL EXPERIMENTS COMPARISON")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    all_results = load_all_experiments()
    
    if not all_results:
        print("ERROR: No experiments found!")
        return
    
    # Generate all plots
    plot_100ep_comparison(all_results)
    plot_all_experiments_heatmap(all_results)
    plot_model_comparison_across_all(all_results)
    plot_epoch_effect(all_results)
    plot_data_and_test_type_effect(all_results)
    create_comprehensive_summary_table(all_results)
    plot_best_models_ranking(all_results)
    plot_cdf_all_100ep(all_results)
    save_results_json(all_results)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Generated 9 comparison plots + JSON summary")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
