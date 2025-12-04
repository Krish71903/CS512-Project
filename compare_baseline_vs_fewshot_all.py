"""
Compare All 3 Models: Baseline vs Few-Shot Trained
Creates comprehensive visualization showing baseline and trained performance
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from pathlib import Path
import seaborn as sns


def parse_baseline_results(txt_path):
    """Parse baseline results from all_baseline_results.txt"""
    models = {}
    
    with open(txt_path, 'r') as f:
        content = f.read()
    
    model_mapping = {
        'RESNET18': 'ResNet18',
        'DENSENET121-MEDICAL': 'DenseNet121',
        'VIT-BASE': 'ViT-Base'
    }
    
    for model_key, display_name in model_mapping.items():
        if model_key in content:
            start = content.find(model_key)
            section_start = content.find('---', start) + 80
            section_end = content.find('\n\n', section_start)
            section = content[section_start:section_end]
            
            results = {}
            pattern = r'3-way (\d+)-shot:\s*([\d.]+)\s*[±�]\s*([\d.]+)'
            matches = re.findall(pattern, section)
            
            for shot, accuracy, ci in matches:
                results[int(shot)] = {
                    'accuracy': float(accuracy),
                    'ci': float(ci)
                }
            
            models[display_name] = results
    
    return models


def parse_trained_csv(csv_path):
    """Parse trained model results from CSV"""
    results = {}
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            scenario = row['Scenario']
            shot = int(scenario.split('-')[0])
            results[shot] = {
                'accuracy': float(row['Accuracy']),
                'ci': float(row['CI'])
            }
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
    return results


def create_comparison_chart(baseline_all, trained_all, save_path):
    """
    Create comprehensive comparison chart
    Shows all 3 models with baseline vs trained
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Model configurations
    models = ['ResNet18', 'DenseNet121', 'ViT-Base']
    colors = {
        'ResNet18': '#2E86AB',
        'DenseNet121': '#A23B72', 
        'ViT-Base': '#F18F01'
    }
    
    # Get all shot values
    all_shots = sorted(set(
        shot for model_results in list(baseline_all.values()) + list(trained_all.values())
        for shot in model_results.keys()
    ))
    
    x_positions = np.arange(len(all_shots))
    width = 0.12  # Width of each bar
    
    # Plot baseline and trained for each model
    for idx, model_name in enumerate(models):
        if model_name not in baseline_all or model_name not in trained_all:
            continue
        
        baseline_results = baseline_all[model_name]
        trained_results = trained_all[model_name]
        
        # Baseline bars (lighter, with pattern)
        baseline_accs = [baseline_results.get(shot, {}).get('accuracy', 0) * 100 for shot in all_shots]
        baseline_cis = [baseline_results.get(shot, {}).get('ci', 0) * 100 for shot in all_shots]
        
        offset_baseline = (idx - 1) * width * 2
        bars_baseline = ax.bar(
            x_positions + offset_baseline,
            baseline_accs,
            width,
            yerr=baseline_cis,
            capsize=4,
            color=colors[model_name],
            alpha=0.5,
            edgecolor='black',
            linewidth=1.5,
            hatch='//',
            label=f'{model_name} Baseline'
        )
        
        # Trained bars (solid)
        trained_accs = [trained_results.get(shot, {}).get('accuracy', 0) * 100 for shot in all_shots]
        trained_cis = [trained_results.get(shot, {}).get('ci', 0) * 100 for shot in all_shots]
        
        offset_trained = offset_baseline + width
        bars_trained = ax.bar(
            x_positions + offset_trained,
            trained_accs,
            width,
            yerr=trained_cis,
            capsize=4,
            color=colors[model_name],
            alpha=0.95,
            edgecolor='black',
            linewidth=1.5,
            label=f'{model_name} Trained'
        )
        
        # Add value labels on trained bars
        for x, acc in zip(x_positions + offset_trained, trained_accs):
            if acc > 0:
                ax.text(x, acc + 3, f'{acc:.1f}',
                       ha='center', va='bottom', fontsize=8,
                       fontweight='bold', color=colors[model_name])
    
    # Random chance line
    random_chance = (1/3) * 100
    ax.axhline(y=random_chance, color='red', linestyle='--',
              linewidth=2, label=f'Random Chance ({random_chance:.1f}%)', alpha=0.7)
    
    # Styling
    ax.set_xlabel('Number of Support Examples (K-shot)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Baseline vs Few-Shot Trained: All Models Comparison\n' +
                'Hatched = Baseline (Frozen) | Solid = Trained (Adapted)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{s}-shot' for s in all_shots], fontsize=12)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=10, loc='upper left', ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {save_path}")
    plt.close()


def create_improvement_chart(baseline_all, trained_all, save_path):
    """
    Create chart showing improvement percentages
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = ['ResNet18', 'DenseNet121', 'ViT-Base']
    colors = {
        'ResNet18': '#2E86AB',
        'DenseNet121': '#A23B72',
        'ViT-Base': '#F18F01'
    }
    
    # Calculate improvements for 5-shot (middle scenario)
    improvements = []
    model_names = []
    bar_colors = []
    
    for model_name in models:
        if model_name in baseline_all and model_name in trained_all:
            baseline = baseline_all[model_name].get(5, {}).get('accuracy', 0)
            trained = trained_all[model_name].get(5, {}).get('accuracy', 0)
            
            if baseline > 0:
                improvement = ((trained - baseline) / baseline) * 100
                improvements.append(improvement)
                model_names.append(model_name)
                bar_colors.append(colors[model_name])
    
    # Create bar chart
    bars = ax.bar(range(len(improvements)), improvements,
                  color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{imp:+.1f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=14, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('Few-Shot Training Improvement over Baseline (5-shot)\n' +
                'Shows how much training helps each model',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def create_summary_table(baseline_all, trained_all):
    """Print a comprehensive summary table"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: BASELINE VS FEW-SHOT TRAINED")
    print("="*100)
    
    models = ['ResNet18', 'DenseNet121', 'ViT-Base']
    shots = [1, 5, 10]
    random_chance = 1/3
    
    for shot in shots:
        print(f"\n{shot}-SHOT RESULTS:")
        print("-"*100)
        print(f"{'Model':<15} {'Baseline':<20} {'Trained':<20} {'Improvement':<20} {'vs Random':<20}")
        print("-"*100)
        
        for model in models:
            if model in baseline_all and model in trained_all:
                baseline_acc = baseline_all[model].get(shot, {}).get('accuracy', 0)
                trained_acc = trained_all[model].get(shot, {}).get('accuracy', 0)
                
                if baseline_acc > 0:
                    improvement = ((trained_acc - baseline_acc) / baseline_acc) * 100
                    vs_random = trained_acc / random_chance
                    
                    print(f"{model:<15} "
                          f"{baseline_acc:.4f} ({baseline_acc/random_chance:.2f}x)    "
                          f"{trained_acc:.4f} ({trained_acc/random_chance:.2f}x)    "
                          f"{improvement:+.1f}%              "
                          f"{vs_random:.2f}x")
        print("-"*100)
    
    print("\n" + "="*100)
    print("AVERAGE PERFORMANCE ACROSS ALL SHOTS:")
    print("="*100)
    print(f"{'Model':<15} {'Avg Baseline':<15} {'Avg Trained':<15} {'Avg Improvement':<15}")
    print("-"*100)
    
    for model in models:
        if model in baseline_all and model in trained_all:
            baseline_accs = [baseline_all[model].get(s, {}).get('accuracy', 0) for s in shots]
            trained_accs = [trained_all[model].get(s, {}).get('accuracy', 0) for s in shots]
            
            avg_baseline = np.mean([a for a in baseline_accs if a > 0])
            avg_trained = np.mean([a for a in trained_accs if a > 0])
            avg_improvement = ((avg_trained - avg_baseline) / avg_baseline) * 100
            
            print(f"{model:<15} {avg_baseline:.4f}         {avg_trained:.4f}         {avg_improvement:+.1f}%")
    
    print("="*100 + "\n")


def main():
    """Create comprehensive baseline vs few-shot comparison"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: ALL 3 MODELS BASELINE VS FEW-SHOT")
    print("="*80)
    
    results_dir = Path("results")
    
    # Load baseline results
    baseline_file = results_dir / "all_baseline_results.txt"
    if not baseline_file.exists():
        print(f"\n[ERROR] Baseline results not found: {baseline_file}")
        print("Run: python evaluate_all_baselines.py")
        return
    
    print("\nLoading baseline results...")
    baseline_all = parse_baseline_results(baseline_file)
    print(f"  Loaded {len(baseline_all)} baseline models")
    
    # Load trained results
    print("\nLoading trained model results...")
    trained_files = {
        'ResNet18': 'chexpert_fast_shot_results.csv',
        'DenseNet121': 'chexpert_chexnet_shot_results.csv',
        'ViT-Base': 'chexpert_vit_shot_results.csv'
    }
    
    trained_all = {}
    for model_name, filename in trained_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            trained_all[model_name] = parse_trained_csv(filepath)
            print(f"  ✓ Loaded {model_name}")
        else:
            print(f"  ✗ Missing {filename}")
    
    if not trained_all:
        print("\n[ERROR] No trained results found!")
        print("Run training scripts first (e.g., python run_chexpert_fast.py)")
        return
    
    # Create visualizations
    print("\n" + "-"*80)
    print("Creating comparison visualizations...")
    print("-"*80)
    
    # 1. Main comparison chart
    create_comparison_chart(
        baseline_all,
        trained_all,
        results_dir / "comparison_all_models_baseline_vs_trained.png"
    )
    
    # 2. Improvement chart
    create_improvement_chart(
        baseline_all,
        trained_all,
        results_dir / "improvement_all_models.png"
    )
    
    # 3. Summary table
    create_summary_table(baseline_all, trained_all)
    
    # Final summary
    print("\n" + "="*80)
    print("SUCCESS! Created comprehensive comparison")
    print("="*80)
    
    print("\nGenerated files:")
    print("  ✓ comparison_all_models_baseline_vs_trained.png")
    print("    → Shows all 3 models: baseline (hatched) vs trained (solid)")
    print("\n  ✓ improvement_all_models.png")
    print("    → Bar chart showing training improvement for each model")
    
    print("\nKey insights from the comparison:")
    print("  • Hatched bars = Baseline (frozen pretrained features)")
    print("  • Solid bars = Trained (domain-adapted features)")
    print("  • Gap between them = Value of few-shot learning")
    print("  • Compare which model benefits most from training")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

