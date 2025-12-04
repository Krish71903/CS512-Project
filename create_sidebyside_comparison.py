"""
Create Side-by-Side Baseline vs Trained Comparison Charts
For each model, show baseline and trained performance on the same plot
"""
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from pathlib import Path


def parse_baseline_results(txt_path):
    """Parse baseline results from all_baseline_results.txt"""
    models = {}
    
    with open(txt_path, 'r') as f:
        content = f.read()
    
    model_names = ['RESNET18', 'DENSENET121-MEDICAL', 'VIT-BASE']
    
    for model_name in model_names:
        if model_name in content:
            start = content.find(model_name)
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
            
            display_name = model_name.replace('-', ' ').title()
            if 'Densenet121' in display_name:
                display_name = 'DenseNet121-Medical'
            if 'Vit' in display_name:
                display_name = 'ViT-Base'
            
            models[display_name] = results
    
    return models


def parse_trained_csv(csv_path):
    """Parse trained model results from CSV"""
    results = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = row['Scenario']
                # Extract shot number (e.g., "1-shot" -> 1)
                shot = int(scenario.split('-')[0])
                results[shot] = {
                    'accuracy': float(row['Accuracy']),
                    'ci': float(row['CI'])
                }
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
    return results


def create_sidebyside_comparison(model_name, baseline_results, trained_results, save_path):
    """
    Create a side-by-side comparison chart for one model
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get shots (use union of both)
    baseline_shots = sorted(baseline_results.keys())
    trained_shots = sorted(trained_results.keys())
    all_shots = sorted(set(baseline_shots + trained_shots))
    
    # Plot baseline (dashed line)
    baseline_accs = [baseline_results.get(shot, {}).get('accuracy') for shot in all_shots]
    baseline_cis = [baseline_results.get(shot, {}).get('ci') for shot in all_shots]
    
    # Filter None values
    baseline_valid_shots = [s for s, a in zip(all_shots, baseline_accs) if a is not None]
    baseline_valid_accs = [a for a in baseline_accs if a is not None]
    baseline_valid_cis = [c for c in baseline_cis if c is not None]
    
    ax.errorbar(baseline_valid_shots, baseline_valid_accs, yerr=baseline_valid_cis,
               marker='o', markersize=10, linewidth=2.5,
               capsize=5, capthick=2,
               color='steelblue', ecolor='steelblue',
               linestyle='--', alpha=0.7,
               label=f'{model_name} Baseline (Frozen)')
    
    # Plot trained (solid line)
    trained_accs = [trained_results.get(shot, {}).get('accuracy') for shot in all_shots]
    trained_cis = [trained_results.get(shot, {}).get('ci') for shot in all_shots]
    
    trained_valid_shots = [s for s, a in zip(all_shots, trained_accs) if a is not None]
    trained_valid_accs = [a for a in trained_accs if a is not None]
    trained_valid_cis = [c for c in trained_cis if c is not None]
    
    ax.errorbar(trained_valid_shots, trained_valid_accs, yerr=trained_valid_cis,
               marker='s', markersize=10, linewidth=3,
               capsize=5, capthick=2,
               color='coral', ecolor='coral',
               linestyle='-', alpha=0.9,
               label=f'{model_name} Trained (Adapted)')
    
    # Add value labels
    for shot, acc in zip(baseline_valid_shots, baseline_valid_accs):
        ax.text(shot, acc - 0.020, f'{acc:.3f}',
               ha='center', va='top', fontsize=9,
               color='steelblue', fontweight='bold')
    
    for shot, acc in zip(trained_valid_shots, trained_valid_accs):
        ax.text(shot, acc + 0.020, f'{acc:.3f}',
               ha='center', va='bottom', fontsize=9,
               color='coral', fontweight='bold')
    
    # Calculate and show improvement
    if len(baseline_valid_shots) > 0 and len(trained_valid_shots) > 0:
        # Use 5-shot for comparison if available, otherwise use middle value
        if 5 in baseline_results and 5 in trained_results:
            baseline_5 = baseline_results[5]['accuracy']
            trained_5 = trained_results[5]['accuracy']
            improvement = (trained_5 - baseline_5) / baseline_5 * 100
            
            # Add improvement annotation
            mid_x = 5
            mid_y = (baseline_5 + trained_5) / 2
            ax.annotate(f'', xy=(mid_x, trained_5), xytext=(mid_x, baseline_5),
                       arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax.text(mid_x + 0.3, mid_y, f'+{improvement:.1f}%\nimprovement',
                   fontsize=10, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Random chance line
    random_chance = 1/3
    ax.axhline(y=random_chance, color='red', linestyle=':',
              linewidth=2, label=f'Random Chance ({random_chance:.3f})', alpha=0.6)
    
    # Styling
    ax.set_xlabel('Number of Support Examples (k-shot)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_name}: Baseline vs Few-Shot Trained Performance\n' +
                'Dashed = Frozen Features | Solid = Domain Adapted',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(all_shots)
    ax.set_ylim([0.3, 0.7])
    
    # Add insights box
    if baseline_valid_accs and trained_valid_accs:
        avg_baseline = np.mean(baseline_valid_accs)
        avg_trained = np.mean(trained_valid_accs)
        avg_improvement = (avg_trained - avg_baseline) / avg_baseline * 100
        
        insight_text = f"Average Improvement: +{avg_improvement:.1f}%\n"
        insight_text += f"Baseline Avg: {avg_baseline:.3f}\n"
        insight_text += f"Trained Avg: {avg_trained:.3f}\n"
        insight_text += f"Training benefit: {avg_trained - avg_baseline:.3f}"
        
        ax.text(0.02, 0.98, insight_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def main():
    """Generate side-by-side comparison charts"""
    
    print("\n" + "="*80)
    print("CREATING SIDE-BY-SIDE BASELINE VS TRAINED CHARTS")
    print("="*80)
    
    results_dir = Path("results")
    
    # Load baseline results
    baseline_file = results_dir / "all_baseline_results.txt"
    if not baseline_file.exists():
        print(f"\n[ERROR] Baseline results not found: {baseline_file}")
        print("Please run: python evaluate_all_baselines.py")
        return
    
    print(f"\nLoading baseline results...")
    all_baseline = parse_baseline_results(baseline_file)
    
    # Model mappings
    model_configs = [
        {
            'name': 'Resnet18',
            'trained_csv': 'chexpert_fast_shot_results.csv',
            'save_name': 'comparison_resnet18'
        },
        {
            'name': 'Densenet121-Medical',
            'trained_csv': 'chexpert_chexnet_shot_results.csv',
            'save_name': 'comparison_densenet121'
        },
        {
            'name': 'Vit-Base',
            'trained_csv': 'chexpert_vit_shot_results.csv',
            'save_name': 'comparison_vit'
        }
    ]
    
    print("\nCreating comparison charts...")
    print("-"*80)
    
    created_charts = []
    
    for config in model_configs:
        model_name = config['name']
        trained_csv = results_dir / config['trained_csv']
        
        # Find baseline results for this model
        baseline_results = None
        for key in all_baseline.keys():
            if key.lower().replace(' ', '').replace('-', '') == model_name.lower().replace('-', ''):
                baseline_results = all_baseline[key]
                break
        
        if baseline_results is None:
            print(f"[WARNING] No baseline results for {model_name}")
            continue
        
        if not trained_csv.exists():
            print(f"[WARNING] Trained results not found: {trained_csv}")
            print(f"           Skipping {model_name}")
            continue
        
        # Load trained results
        trained_results = parse_trained_csv(trained_csv)
        
        if not trained_results:
            print(f"[WARNING] Could not parse trained results for {model_name}")
            continue
        
        # Create comparison chart
        save_path = results_dir / f"{config['save_name']}_baseline_vs_trained.png"
        create_sidebyside_comparison(model_name, baseline_results, trained_results, save_path)
        created_charts.append(save_path.name)
    
    # Summary
    print("\n" + "="*80)
    print("SUCCESS! Generated side-by-side comparison charts")
    print("="*80)
    
    if created_charts:
        print("\nGenerated files:")
        for chart in created_charts:
            print(f"  ✓ {chart}")
        
        print("\nThese charts show:")
        print("  • Baseline (dashed) vs Trained (solid) on same plot")
        print("  • Improvement arrows and percentages")
        print("  • Direct visual comparison of training benefit")
        
        print("\nUse these for:")
        print("  • Paper figures showing training improvement")
        print("  • Presentations demonstrating few-shot learning value")
        print("  • Analysis of which models benefit most from training")
    else:
        print("\n[WARNING] No charts were created")
        print("Make sure you have run:")
        print("  1. python evaluate_all_baselines.py")
        print("  2. python run_chexpert_fast.py (or other training scripts)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

