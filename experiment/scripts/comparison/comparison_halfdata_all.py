#!/usr/bin/env python3
"""
Half Data Experiments Comparison
================================

Compares all half data experiments:
1. 30 epochs, LR=0.0001, Half Data
2. 50 epochs, LR=0.0001, Half Data
3. 100 epochs, LR=0.0001, Half Data

And separately the augmented test experiments:
4. 30 epochs, LR=0.0001, Half Data + Augmented Test
5. 50 epochs, LR=0.0001, Half Data + Augmented Test

Output: testing/comparison_halfdata_all/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


MODELS = ["ResNet18", "EfficientNet", "ConvNeXt"]
MODEL_COLORS = {"ResNet18": "#e41a1c", "EfficientNet": "#377eb8", "ConvNeXt": "#4daf4a"}

HALF_DATA_EXPERIMENTS = {
    "30ep Half": {
        "path": Path("testing/30epochs_lr0.0001_halfdata"),
        "color": "#2ca02c",
        "epochs": 30,
    },
    "50ep Half": {
        "path": Path("testing/50ep_lr0001_halfdata"),
        "color": "#17becf",
        "epochs": 50,
    },
    "100ep Half": {
        "path": Path("testing/100ep_lr0001_halfdata"),
        "color": "#e377c2",
        "epochs": 100,
    },
}

AUGMENTED_EXPERIMENTS = {
    "30ep Half+Aug": {
        "path": Path("testing/30ep_lr0001_halfdata_augtest"),
        "color": "#9467bd",
        "epochs": 30,
    },
    "50ep Half+Aug": {
        "path": Path("testing/50ep_lr0001_halfdata_augtest"),
        "color": "#bcbd22",
        "epochs": 50,
    },
}

OUTPUT_DIR = Path("testing/comparison_halfdata_all")


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
    """Generate comparison plots for half data experiments."""
    print("=" * 70)
    print("HALF DATA EXPERIMENTS COMPARISON")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load half data experiments
    print("\nLoading Half Data experiments...")
    half_results = {}
    for exp_name, exp_config in HALF_DATA_EXPERIMENTS.items():
        print(f"  Loading {exp_name}...")
        half_results[exp_name] = load_experiment_results(exp_config["path"])
    
    # Load augmented experiments
    print("\nLoading Augmented Test experiments...")
    aug_results = {}
    for exp_name, exp_config in AUGMENTED_EXPERIMENTS.items():
        print(f"  Loading {exp_name}...")
        aug_results[exp_name] = load_experiment_results(exp_config["path"])
    
    half_exp_names = list(HALF_DATA_EXPERIMENTS.keys())
    aug_exp_names = list(AUGMENTED_EXPERIMENTS.keys())
    
    # 1. Half Data Summary Table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ['Model'] + [f'{exp}\nMean' for exp in half_exp_names] + [f'{exp}\n<10m' for exp in half_exp_names]
    rows = []
    
    for model in MODELS:
        row = [model]
        for exp_name in half_exp_names:
            if model in half_results[exp_name]:
                row.append(f"{half_results[exp_name][model]['mean_error']:.2f}m")
            else:
                row.append("N/A")
        for exp_name in half_exp_names:
            if model in half_results[exp_name]:
                row.append(f"{half_results[exp_name][model]['within_10m']:.1f}%")
            else:
                row.append("N/A")
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, model in enumerate(MODELS):
        table[(i+1, 0)].set_facecolor(MODEL_COLORS[model])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Half Data Experiments (LR=0.0001, 50% Data, Regular Test)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_half_data_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ 01_half_data_summary.png")
    
    # 2. Augmented Test Summary Table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ['Model'] + [f'{exp}\nMean' for exp in aug_exp_names] + [f'{exp}\n<10m' for exp in aug_exp_names]
    rows = []
    
    for model in MODELS:
        row = [model]
        for exp_name in aug_exp_names:
            if model in aug_results[exp_name]:
                row.append(f"{aug_results[exp_name][model]['mean_error']:.2f}m")
            else:
                row.append("N/A")
        for exp_name in aug_exp_names:
            if model in aug_results[exp_name]:
                row.append(f"{aug_results[exp_name][model]['within_10m']:.1f}%")
            else:
                row.append("N/A")
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#9467bd')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, model in enumerate(MODELS):
        table[(i+1, 0)].set_facecolor(MODEL_COLORS[model])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Augmented Test Experiments (LR=0.0001, 50% Data, Augmented Test)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_augmented_test_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_augmented_test_summary.png")
    
    # 3. Half Data Error Evolution
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = [HALF_DATA_EXPERIMENTS[exp]["epochs"] for exp in half_exp_names]
    
    for model in MODELS:
        means = []
        for exp_name in half_exp_names:
            if model in half_results[exp_name]:
                means.append(half_results[exp_name][model]['mean_error'])
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
    ax.set_title("Half Data: Error Evolution\n(LR=0.0001, Regular Test)", fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_half_data_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_half_data_evolution.png")
    
    # 4. Augmented Test Error Evolution
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = [AUGMENTED_EXPERIMENTS[exp]["epochs"] for exp in aug_exp_names]
    
    for model in MODELS:
        means = []
        for exp_name in aug_exp_names:
            if model in aug_results[exp_name]:
                means.append(aug_results[exp_name][model]['mean_error'])
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
    ax.set_title("Augmented Test: Error Evolution\n(LR=0.0001, Augmented Test)", fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_augmented_test_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_augmented_test_evolution.png")
    
    # 5. Combined Bar Chart - Half Data vs Augmented
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(MODELS))
    width = 0.35
    
    # 30 epochs comparison
    ax = axes[0]
    half_means = [half_results["30ep Half"].get(m, {}).get('mean_error', np.nan) for m in MODELS]
    aug_means = [aug_results["30ep Half+Aug"].get(m, {}).get('mean_error', np.nan) for m in MODELS]
    
    bars1 = ax.bar(x - width/2, half_means, width, label='Regular Test', color=HALF_DATA_EXPERIMENTS["30ep Half"]["color"])
    bars2 = ax.bar(x + width/2, aug_means, width, label='Augmented Test', color=AUGMENTED_EXPERIMENTS["30ep Half+Aug"]["color"])
    
    for bar, val in zip(bars1, half_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, aug_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title("30 Epochs", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 50 epochs comparison
    ax = axes[1]
    half_means = [half_results["50ep Half"].get(m, {}).get('mean_error', np.nan) for m in MODELS]
    aug_means = [aug_results["50ep Half+Aug"].get(m, {}).get('mean_error', np.nan) for m in MODELS]
    
    bars1 = ax.bar(x - width/2, half_means, width, label='Regular Test', color=HALF_DATA_EXPERIMENTS["50ep Half"]["color"])
    bars2 = ax.bar(x + width/2, aug_means, width, label='Augmented Test', color=AUGMENTED_EXPERIMENTS["50ep Half+Aug"]["color"])
    
    for bar, val in zip(bars1, half_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, aug_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title("50 Epochs", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Regular Test vs Augmented Test\n(Half Data, LR=0.0001)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_regular_vs_augmented.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_regular_vs_augmented.png")
    
    # 6. All Half-Data CDFs
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Regular Test
    for ax, model in zip(axes[0], MODELS):
        for exp_name, exp_config in HALF_DATA_EXPERIMENTS.items():
            if model in half_results[exp_name]:
                errors = np.sort(half_results[exp_name][model]['errors'])
                cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
                ax.plot(errors, cdf, linewidth=2, label=exp_name, color=exp_config["color"])
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel("Error (meters)", fontsize=10)
        ax.set_ylabel("Cumulative %", fontsize=10)
        ax.set_title(f"{model}\n(Regular Test)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
    
    # Row 2: Augmented Test
    for ax, model in zip(axes[1], MODELS):
        for exp_name, exp_config in AUGMENTED_EXPERIMENTS.items():
            if model in aug_results[exp_name]:
                errors = np.sort(aug_results[exp_name][model]['errors'])
                cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
                ax.plot(errors, cdf, linewidth=2, label=exp_name, color=exp_config["color"])
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel("Error (meters)", fontsize=10)
        ax.set_ylabel("Cumulative %", fontsize=10)
        ax.set_title(f"{model}\n(Augmented Test)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 80)
    
    plt.suptitle("Error CDF: Half Data Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_cdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_cdf_comparison.png")
    
    # 7. Improvement Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Half Data: 30ep -> 50ep improvement
    ax = axes[0]
    improvements = []
    for model in MODELS:
        if model in half_results["30ep Half"] and model in half_results["50ep Half"]:
            imp = half_results["30ep Half"][model]['mean_error'] - half_results["50ep Half"][model]['mean_error']
            improvements.append(imp)
        else:
            improvements.append(0)
    
    bars = ax.bar(MODELS, improvements, color=[MODEL_COLORS[m] for m in MODELS])
    for bar, val in zip(bars, improvements):
        color = 'green' if val > 0 else 'red'
        offset = 0.2 if val > 0 else -1.0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
               f'{val:+.2f}m', ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel("Error Reduction (meters)", fontsize=11)
    ax.set_title("Half Data: 30ep → 50ep\n(positive = better)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Augmented Test: 30ep -> 50ep improvement
    ax = axes[1]
    improvements = []
    for model in MODELS:
        if model in aug_results["30ep Half+Aug"] and model in aug_results["50ep Half+Aug"]:
            imp = aug_results["30ep Half+Aug"][model]['mean_error'] - aug_results["50ep Half+Aug"][model]['mean_error']
            improvements.append(imp)
        else:
            improvements.append(0)
    
    bars = ax.bar(MODELS, improvements, color=[MODEL_COLORS[m] for m in MODELS])
    for bar, val in zip(bars, improvements):
        color = 'green' if val > 0 else 'red'
        offset = 0.2 if val > 0 else -1.0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
               f'{val:+.2f}m', ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel("Error Reduction (meters)", fontsize=11)
    ax.set_title("Augmented Test: 30ep → 50ep\n(positive = better)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Error Improvement Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_improvement_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_improvement_analysis.png")
    
    # 8. Combined Overview - All 4 experiments
    fig, ax = plt.subplots(figsize=(16, 8))
    
    all_exp = list(HALF_DATA_EXPERIMENTS.keys()) + list(AUGMENTED_EXPERIMENTS.keys())
    all_configs = {**HALF_DATA_EXPERIMENTS, **AUGMENTED_EXPERIMENTS}
    all_data = {**half_results, **aug_results}
    
    x = np.arange(len(MODELS))
    width = 0.2
    
    for i, exp_name in enumerate(all_exp):
        means = [all_data[exp_name].get(m, {}).get('mean_error', np.nan) for m in MODELS]
        bars = ax.bar(x + i * width, means, width, label=exp_name, color=all_configs[exp_name]["color"])
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{val:.1f}', ha='center', fontsize=8, rotation=90)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Error (meters)", fontsize=12)
    ax.set_title("All Half Data Experiments Comparison\n(LR=0.0001, 50% Data)", fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_all_halfdata_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_all_halfdata_overview.png")
    
    # Save summary text
    with open(OUTPUT_DIR / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HALF DATA EXPERIMENTS COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HALF DATA (Regular Test) - LR=0.0001, 50% Data\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<15}")
        for exp in half_exp_names:
            f.write(f"{exp:<15}")
        f.write("\n")
        f.write("-" * 60 + "\n")
        
        for model in MODELS:
            f.write(f"{model:<15}")
            for exp_name in half_exp_names:
                if model in half_results[exp_name]:
                    val = half_results[exp_name][model]['mean_error']
                    f.write(f"{val:<15.2f}")
                else:
                    f.write(f"{'N/A':<15}")
            f.write("\n")
        
        f.write("\n\nAUGMENTED TEST - LR=0.0001, 50% Data\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<15}")
        for exp in aug_exp_names:
            f.write(f"{exp:<15}")
        f.write("\n")
        f.write("-" * 60 + "\n")
        
        for model in MODELS:
            f.write(f"{model:<15}")
            for exp_name in aug_exp_names:
                if model in aug_results[exp_name]:
                    val = aug_results[exp_name][model]['mean_error']
                    f.write(f"{val:<15.2f}")
                else:
                    f.write(f"{'N/A':<15}")
            f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
        
        # Best overall in half data
        best_model = None
        best_exp = None
        best_error = float('inf')
        for exp_name in half_exp_names:
            for model in MODELS:
                if model in half_results[exp_name]:
                    if half_results[exp_name][model]['mean_error'] < best_error:
                        best_error = half_results[exp_name][model]['mean_error']
                        best_model = model
                        best_exp = exp_name
        
        f.write(f"🏆 BEST HALF DATA (Regular): {best_model} in {best_exp} - {best_error:.2f}m\n")
        
        best_error = float('inf')
        for exp_name in aug_exp_names:
            for model in MODELS:
                if model in aug_results[exp_name]:
                    if aug_results[exp_name][model]['mean_error'] < best_error:
                        best_error = aug_results[exp_name][model]['mean_error']
                        best_model = model
                        best_exp = exp_name
        
        f.write(f"🏆 BEST AUGMENTED TEST: {best_model} in {best_exp} - {best_error:.2f}m\n")
        f.write("=" * 70 + "\n")
    
    print("  ✓ comparison_summary.txt")
    print(f"\n✅ Saved to: {OUTPUT_DIR}/")


def main():
    create_comparison_plots()


if __name__ == "__main__":
    main()
