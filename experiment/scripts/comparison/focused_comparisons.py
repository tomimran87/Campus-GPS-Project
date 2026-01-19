#!/usr/bin/env python3
"""
Focused Comparisons:
1. 50ep Full (LR=0.001) vs 50ep Half (LR=0.0001)
2. 30ep Half (LR=0.0001) vs 50ep Half (LR=0.0001)

These targeted comparisons help understand:
- Effect of data size and LR at 50 epochs
- Effect of more epochs with same data/LR setup
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]
MODEL_COLORS = {"ResNet18": "#e41a1c", "EfficientNet": "#377eb8", "ConvNeXt": "#4daf4a"}

# Define the experiments to compare
COMPARISONS = {
    "50ep_comparison": {
        "title": "50 Epochs: Full Data (LR=0.001) vs Half Data (LR=0.0001)",
        "exp1": {
            "path": Path("testing/50ep_lr001_fulldata"),
            "label": "50ep Full (LR=0.001)",
            "color": "#d62728",
        },
        "exp2": {
            "path": Path("testing/50ep_lr0001_halfdata"),
            "label": "50ep Half (LR=0.0001)",
            "color": "#9467bd",
        },
        "output": Path("testing/comparison_50ep"),
    },
    "half_data_epochs": {
        "title": "Half Data: 30 Epochs vs 50 Epochs (LR=0.0001)",
        "exp1": {
            "path": Path("testing/30epochs_lr0.0001_halfdata"),
            "label": "30ep Half (LR=0.0001)",
            "color": "#ff7f0e",
        },
        "exp2": {
            "path": Path("testing/50ep_lr0001_halfdata"),
            "label": "50ep Half (LR=0.0001)",
            "color": "#9467bd",
        },
        "output": Path("testing/comparison_halfdata_epochs"),
    },
    "augtest_epochs": {
        "title": "Augmented Test: 30 Epochs vs 50 Epochs (LR=0.0001)",
        "exp1": {
            "path": Path("testing/30ep_lr0001_halfdata_augtest"),
            "label": "30ep Half+Aug",
            "color": "#2ca02c",
        },
        "exp2": {
            "path": Path("testing/50ep_lr0001_halfdata_augtest"),
            "label": "50ep Half+Aug",
            "color": "#8c564b",
        },
        "output": Path("testing/comparison_augtest_epochs"),
    },
}


def load_experiment_results(exp_path):
    """Load results for all models in an experiment."""
    results = {}
    
    for model in MODELS:
        model_dir = exp_path / model
        errors_path = model_dir / f"{model.lower()}_errors.npy"
        
        if not errors_path.exists():
            print(f"  ⚠ {model} not found at {errors_path}")
            continue
        
        errors = np.load(errors_path)
        
        with open(model_dir / "test_results.json", 'r') as f:
            test_results = json.load(f)
        
        results[model] = {
            "errors": errors,
            **test_results
        }
    
    return results


def create_comparison_plots(comp_name, comp_config):
    """Generate comparison plots for two experiments."""
    print(f"\n{'='*60}")
    print(f"Comparison: {comp_config['title']}")
    print(f"{'='*60}")
    
    output_dir = comp_config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load both experiments
    print(f"Loading {comp_config['exp1']['label']}...")
    results1 = load_experiment_results(comp_config["exp1"]["path"])
    print(f"Loading {comp_config['exp2']['label']}...")
    results2 = load_experiment_results(comp_config["exp2"]["path"])
    
    exp1_label = comp_config["exp1"]["label"]
    exp2_label = comp_config["exp2"]["label"]
    exp1_color = comp_config["exp1"]["color"]
    exp2_color = comp_config["exp2"]["color"]
    
    # 1. Summary Table
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    columns = ['Model', f'{exp1_label}\nMean', f'{exp2_label}\nMean', 'Diff',
               f'{exp1_label}\n<10m', f'{exp2_label}\n<10m', 'Diff']
    rows = []
    
    for model in MODELS:
        if model in results1 and model in results2:
            r1, r2 = results1[model], results2[model]
            mean_diff = r2['mean_error'] - r1['mean_error']
            acc_diff = r2['within_10m'] - r1['within_10m']
            rows.append([
                model,
                f"{r1['mean_error']:.2f}m",
                f"{r2['mean_error']:.2f}m",
                f"{mean_diff:+.2f}m",
                f"{r1['within_10m']:.1f}%",
                f"{r2['within_10m']:.1f}%",
                f"{acc_diff:+.1f}%",
            ])
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, model in enumerate(MODELS):
        if model in results1:
            table[(i+1, 0)].set_facecolor(MODEL_COLORS[model])
            table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f"{comp_config['title']}\nComparison Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "01_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_summary_table.png")
    
    # 2. Mean Error Bar Chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(MODELS))
    width = 0.35
    
    means1 = [results1[m]['mean_error'] if m in results1 else 0 for m in MODELS]
    means2 = [results2[m]['mean_error'] if m in results2 else 0 for m in MODELS]
    
    bars1 = ax.bar(x - width/2, means1, width, label=exp1_label, color=exp1_color)
    bars2 = ax.bar(x + width/2, means2, width, label=exp2_label, color=exp2_color)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{height:.1f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title(f"{comp_config['title']}\nMean Error Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "02_mean_error_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_mean_error_comparison.png")
    
    # 3. Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    acc1 = [results1[m]['within_10m'] if m in results1 else 0 for m in MODELS]
    acc2 = [results2[m]['within_10m'] if m in results2 else 0 for m in MODELS]
    
    bars1 = ax.bar(x - width/2, acc1, width, label=exp1_label, color=exp1_color)
    bars2 = ax.bar(x + width/2, acc2, width, label=exp2_label, color=exp2_color)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy Within 10m (%)", fontsize=12)
    ax.set_title(f"{comp_config['title']}\nAccuracy Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "03_accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_accuracy_comparison.png")
    
    # 4. CDF Comparison per model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, model in zip(axes, MODELS):
        if model in results1:
            errors1 = np.sort(results1[model]['errors'])
            cdf1 = np.arange(1, len(errors1) + 1) / len(errors1) * 100
            ax.plot(errors1, cdf1, color=exp1_color, linewidth=2, label=exp1_label)
        
        if model in results2:
            errors2 = np.sort(results2[model]['errors'])
            cdf2 = np.arange(1, len(errors2) + 1) / len(errors2) * 100
            ax.plot(errors2, cdf2, color=exp2_color, linewidth=2, label=exp2_label)
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='10m')
        
        ax.set_xlabel("Error (meters)", fontsize=11)
        ax.set_ylabel("Cumulative %", fontsize=11)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
    
    plt.suptitle(f"{comp_config['title']}\nError CDF by Model", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "04_cdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_cdf_comparison.png")
    
    # 5. Boxplot Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    
    positions = []
    data = []
    colors = []
    labels = []
    
    for i, model in enumerate(MODELS):
        if model in results1:
            positions.append(i * 3)
            data.append(results1[model]['errors'])
            colors.append(exp1_color)
            labels.append(f"{model}\n{exp1_label}")
        
        if model in results2:
            positions.append(i * 3 + 1)
            data.append(results2[model]['errors'])
            colors.append(exp2_color)
            labels.append(f"{model}\n{exp2_label}")
    
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.7)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel("Error (meters)", fontsize=12)
    ax.set_title(f"{comp_config['title']}\nError Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "05_boxplot_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_boxplot_comparison.png")
    
    # 6. Improvement Chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    improvements = []
    for model in MODELS:
        if model in results1 and model in results2:
            # Improvement = exp1 - exp2 (positive means exp2 is better)
            improvements.append(results1[model]['mean_error'] - results2[model]['mean_error'])
    
    bars = ax.bar(MODELS, improvements, color=[MODEL_COLORS[m] for m in MODELS], edgecolor='white')
    
    for bar, val in zip(bars, improvements):
        color = 'green' if val > 0 else 'red'
        sign = '+' if val > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val > 0 else -0.8),
               f'{sign}{val:.2f}m', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(f"Improvement: {exp1_label} → {exp2_label} (meters)", fontsize=11)
    ax.set_title(f"{comp_config['title']}\nError Improvement (positive = {exp2_label} is better)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "06_improvement_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_improvement_chart.png")
    
    # 7. Comprehensive Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('mean_error', 'Mean Error (m)', False),
        ('median_error', 'Median Error (m)', False),
        ('within_10m', 'Within 10m (%)', True),
        ('within_20m', 'Within 20m (%)', True),
    ]
    
    for ax, (metric, label, higher_is_better) in zip(axes.flatten(), metrics):
        vals1 = [results1[m][metric] if m in results1 else 0 for m in MODELS]
        vals2 = [results2[m][metric] if m in results2 else 0 for m in MODELS]
        
        bars1 = ax.bar(x - width/2, vals1, width, label=exp1_label, color=exp1_color)
        bars2 = ax.bar(x + width/2, vals2, width, label=exp2_label, color=exp2_color)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(label, fontsize=12, fontweight='bold')
    
    plt.suptitle(f"{comp_config['title']}\nComprehensive Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_comprehensive_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_comprehensive_metrics.png")
    
    # Save text summary
    with open(output_dir / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{comp_config['title']}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<15} {'Exp1 Mean':<12} {'Exp2 Mean':<12} {'Diff':<10} {'Winner':<15}\n")
        f.write("-" * 60 + "\n")
        
        for model in MODELS:
            if model in results1 and model in results2:
                r1, r2 = results1[model], results2[model]
                diff = r2['mean_error'] - r1['mean_error']
                winner = exp1_label if r1['mean_error'] < r2['mean_error'] else exp2_label
                f.write(f"{model:<15} {r1['mean_error']:<12.2f} {r2['mean_error']:<12.2f} "
                       f"{diff:+10.2f} {winner:<15}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Exp1: {exp1_label}\n")
        f.write(f"Exp2: {exp2_label}\n")
        f.write("=" * 70 + "\n")
    
    print(f"  ✓ comparison_summary.txt")
    print(f"\n✅ Saved to: {output_dir}/")


def main():
    print("=" * 70)
    print("FOCUSED EXPERIMENT COMPARISONS")
    print("=" * 70)
    
    for comp_name, comp_config in COMPARISONS.items():
        create_comparison_plots(comp_name, comp_config)
    
    print("\n" + "=" * 70)
    print("ALL COMPARISONS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
