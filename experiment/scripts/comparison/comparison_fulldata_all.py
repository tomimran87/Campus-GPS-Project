#!/usr/bin/env python3
"""
Full Data Experiments Comparison
================================

Compares all full data experiments:
1. 30 epochs, LR=0.001, Full Data
2. 50 epochs, LR=0.001, Full Data  
3. 100 epochs, LR=0.001, Full Data

Output: testing/comparison_fulldata_all/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]
MODEL_COLORS = {"ResNet18": "#e41a1c", "EfficientNet": "#377eb8", "ConvNeXt": "#4daf4a"}

EXPERIMENTS = {
    "30ep Full": {
        "path": Path("testing/30epochs"),
        "color": "#1f77b4",
        "epochs": 30,
    },
    "50ep Full": {
        "path": Path("testing/50ep_lr001_fulldata"),
        "color": "#d62728",
        "epochs": 50,
    },
    "100ep Full": {
        "path": Path("testing/100ep_lr001_fulldata"),
        "color": "#e377c2",
        "epochs": 100,
    },
}

OUTPUT_DIR = Path("testing/comparison_fulldata_all")


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


def create_comparison_plots():
    """Generate comparison plots for full data experiments."""
    print("=" * 70)
    print("FULL DATA EXPERIMENTS COMPARISON")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"Loading {exp_name}...")
        all_results[exp_name] = load_experiment_results(exp_config["path"])
    
    exp_names = list(EXPERIMENTS.keys())
    
    # 1. Summary Table
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')
    
    columns = ['Model'] + [f'{exp}\nMean' for exp in exp_names] + [f'{exp}\n<10m' for exp in exp_names]
    rows = []
    
    for model in MODELS:
        row = [model]
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                row.append(f"{all_results[exp_name][model]['mean_error']:.2f}m")
            else:
                row.append("N/A")
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                row.append(f"{all_results[exp_name][model]['within_10m']:.1f}%")
            else:
                row.append("N/A")
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, model in enumerate(MODELS):
        table[(i+1, 0)].set_facecolor(MODEL_COLORS[model])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Full Data Experiments Comparison\n(LR=0.001, 100% Data)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_summary_table.png")
    
    # 2. Mean Error Evolution by Epochs
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = [EXPERIMENTS[exp]["epochs"] for exp in exp_names]
    
    for model in MODELS:
        means = []
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                means.append(all_results[exp_name][model]['mean_error'])
            else:
                means.append(np.nan)
        
        ax.plot(epochs, means, marker='o', markersize=10, linewidth=2, 
                label=model, color=MODEL_COLORS[model])
        
        for i, (ep, mean) in enumerate(zip(epochs, means)):
            if not np.isnan(mean):
                ax.annotate(f'{mean:.1f}m', (ep, mean), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title("Mean Error Evolution with More Epochs\n(Full Data, LR=0.001)", fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_error_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_error_evolution.png")
    
    # 3. Accuracy Evolution
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in MODELS:
        accs = []
        for exp_name in exp_names:
            if model in all_results[exp_name]:
                accs.append(all_results[exp_name][model]['within_10m'])
            else:
                accs.append(np.nan)
        
        ax.plot(epochs, accs, marker='s', markersize=10, linewidth=2, 
                label=model, color=MODEL_COLORS[model])
        
        for i, (ep, acc) in enumerate(zip(epochs, accs)):
            if not np.isnan(acc):
                ax.annotate(f'{acc:.1f}%', (ep, acc), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Accuracy Within 10m (%)", fontsize=12)
    ax.set_title("Accuracy Evolution with More Epochs\n(Full Data, LR=0.001)", fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_accuracy_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_accuracy_evolution.png")
    
    # 4. Bar Chart Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(MODELS))
    width = 0.25
    
    # Mean Error
    ax = axes[0]
    for i, exp_name in enumerate(exp_names):
        means = [all_results[exp_name].get(m, {}).get('mean_error', np.nan) for m in MODELS]
        bars = ax.bar(x + i * width, means, width, label=exp_name, color=EXPERIMENTS[exp_name]["color"])
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title("Mean Error Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Accuracy
    ax = axes[1]
    for i, exp_name in enumerate(exp_names):
        accs = [all_results[exp_name].get(m, {}).get('within_10m', np.nan) for m in MODELS]
        bars = ax.bar(x + i * width, accs, width, label=exp_name, color=EXPERIMENTS[exp_name]["color"])
        for bar, val in zip(bars, accs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Within 10m (%)", fontsize=12)
    ax.set_title("Accuracy Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Full Data Experiments: 30 vs 50 vs 100 Epochs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_bar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_bar_comparison.png")
    
    # 5. CDF Comparison per model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, model in zip(axes, MODELS):
        for exp_name, exp_config in EXPERIMENTS.items():
            if model in all_results[exp_name]:
                errors = np.sort(all_results[exp_name][model]['errors'])
                cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
                ax.plot(errors, cdf, linewidth=2, label=exp_name, color=exp_config["color"])
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel("Error (meters)", fontsize=11)
        ax.set_ylabel("Cumulative %", fontsize=11)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
    
    plt.suptitle("Error CDF: Full Data Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_cdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_cdf_comparison.png")
    
    # 6. Improvement Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 30ep -> 50ep improvement
    ax = axes[0]
    improvements = []
    for model in MODELS:
        if model in all_results["30ep Full"] and model in all_results["50ep Full"]:
            imp = all_results["30ep Full"][model]['mean_error'] - all_results["50ep Full"][model]['mean_error']
            improvements.append(imp)
        else:
            improvements.append(0)
    
    bars = ax.bar(MODELS, improvements, color=[MODEL_COLORS[m] for m in MODELS])
    for bar, val in zip(bars, improvements):
        color = 'green' if val > 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if val > 0 else -1.5),
               f'{val:+.2f}m', ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel("Error Reduction (meters)", fontsize=11)
    ax.set_title("Improvement: 30ep → 50ep\n(positive = better)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 50ep -> 100ep improvement
    ax = axes[1]
    improvements = []
    for model in MODELS:
        if model in all_results["50ep Full"] and model in all_results["100ep Full"]:
            imp = all_results["50ep Full"][model]['mean_error'] - all_results["100ep Full"][model]['mean_error']
            improvements.append(imp)
        else:
            improvements.append(0)
    
    bars = ax.bar(MODELS, improvements, color=[MODEL_COLORS[m] for m in MODELS])
    for bar, val in zip(bars, improvements):
        color = 'green' if val > 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if val > 0 else -1.5),
               f'{val:+.2f}m', ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel("Error Reduction (meters)", fontsize=11)
    ax.set_title("Improvement: 50ep → 100ep\n(positive = better)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Error Improvement Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_improvement_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_improvement_analysis.png")
    
    # 7. Boxplot Comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    
    positions = []
    data = []
    colors = []
    labels = []
    
    pos = 0
    for model in MODELS:
        for exp_name, exp_config in EXPERIMENTS.items():
            if model in all_results[exp_name]:
                positions.append(pos)
                data.append(all_results[exp_name][model]['errors'])
                colors.append(exp_config["color"])
                labels.append(f"{model}\n{exp_name}")
                pos += 1
        pos += 0.5  # Gap between models
    
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.7)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel("Error (meters)", fontsize=12)
    ax.set_title("Error Distribution: Full Data Experiments", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 80)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_boxplot_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_boxplot_comparison.png")
    
    # Save summary text
    with open(OUTPUT_DIR / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FULL DATA EXPERIMENTS COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("All experiments: LR=0.001, 100% Data, Regular Test\n\n")
        
        f.write("RESULTS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<15}")
        for exp in exp_names:
            f.write(f"{exp:<15}")
        f.write("Best\n")
        f.write("-" * 60 + "\n")
        
        for model in MODELS:
            f.write(f"{model:<15}")
            best_exp = None
            best_val = float('inf')
            for exp_name in exp_names:
                if model in all_results[exp_name]:
                    val = all_results[exp_name][model]['mean_error']
                    f.write(f"{val:<15.2f}")
                    if val < best_val:
                        best_val = val
                        best_exp = exp_name
                else:
                    f.write(f"{'N/A':<15}")
            f.write(f"{best_exp}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        
        # Best overall
        best_model = None
        best_exp = None
        best_error = float('inf')
        for exp_name in exp_names:
            for model in MODELS:
                if model in all_results[exp_name]:
                    if all_results[exp_name][model]['mean_error'] < best_error:
                        best_error = all_results[exp_name][model]['mean_error']
                        best_model = model
                        best_exp = exp_name
        
        f.write(f"🏆 BEST OVERALL: {best_model} in {best_exp}\n")
        f.write(f"   Mean Error: {best_error:.2f}m\n")
        f.write("=" * 70 + "\n")
    
    print("  ✓ comparison_summary.txt")
    print(f"\n✅ Saved to: {OUTPUT_DIR}/")


def main():
    create_comparison_plots()


if __name__ == "__main__":
    main()
