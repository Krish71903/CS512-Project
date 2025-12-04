"""
Evaluate Baselines for Multiple Backbones
Tests frozen pretrained features with Prototypical Networks
- ResNet18
- DenseNet121 (Medical/CheXNet)
- ViT-Base
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import BackboneFactory


class FrozenPrototypicalBaseline(nn.Module):
    """
    Baseline using frozen pretrained features with Prototypical Networks
    No training - just tests how well pretrained features work
    """
    def __init__(self, backbone, embedding_dim):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()


class BaselineEvaluator:
    """Evaluator for frozen baseline models"""
    
    def __init__(self, model, config, dataset, model_name="Model"):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.dataset = dataset
        self.device = config.DEVICE
        self.model_name = model_name
        self.model.eval()
    
    def evaluate_shots(self, n_way=3, k_shots=[1, 5, 10], n_query=10, n_episodes=100):
        """
        Evaluate baseline across different shot scenarios
        """
        print(f"\n{'='*80}")
        print(f"BASELINE EVALUATION: {self.model_name.upper()}")
        print(f"{'='*80}")
        print(f"Configuration: {n_way}-way classification")
        print(f"Query samples per class: {n_query}")
        print(f"Episodes per scenario: {n_episodes}")
        print(f"Note: Frozen pretrained features with Prototypical Networks")
        print(f"{'='*80}\n")
        
        results = {}
        
        for k_shot in k_shots:
            print(f"\nEvaluating {k_shot}-shot scenario...")
            accuracies = []
            
            with torch.no_grad():
                for _ in tqdm(range(n_episodes), desc=f"  {k_shot}-shot"):
                    # Sample episode
                    support_images, support_labels, query_images, query_labels = \
                        self.dataset.sample_episode(n_way, k_shot, n_query)
                    
                    # Move to device
                    support_images = support_images.to(self.device)
                    support_labels = support_labels.to(self.device)
                    query_images = query_images.to(self.device)
                    query_labels = query_labels.to(self.device)
                    
                    # Extract features using frozen pretrained backbone
                    support_features = self.model.backbone(support_images)
                    query_features = self.model.backbone(query_images)
                    
                    # Flatten features if needed
                    if len(support_features.shape) > 2:
                        support_features = support_features.view(support_features.size(0), -1)
                    if len(query_features.shape) > 2:
                        query_features = query_features.view(query_features.size(0), -1)
                    
                    # Compute class prototypes (mean of support features per class)
                    prototypes = []
                    for label in range(n_way):
                        class_mask = (support_labels == label)
                        class_features = support_features[class_mask]
                        prototype = class_features.mean(dim=0)
                        prototypes.append(prototype)
                    
                    prototypes = torch.stack(prototypes)  # [n_way, embedding_dim]
                    
                    # Compute distances (Euclidean)
                    query_expanded = query_features.unsqueeze(1)  # [n_query, 1, embed_dim]
                    proto_expanded = prototypes.unsqueeze(0)  # [1, n_way, embed_dim]
                    distances = torch.sum((query_expanded - proto_expanded) ** 2, dim=2)
                    
                    # Predict (nearest prototype)
                    predictions = torch.argmin(distances, dim=1)
                    accuracy = (predictions == query_labels).float().mean().item()
                    accuracies.append(accuracy)
            
            # Compute statistics
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            ci = 1.96 * std_acc / np.sqrt(n_episodes)
            
            results[f"{k_shot}-shot"] = {
                'accuracy': avg_acc,
                'std': std_acc,
                'ci': ci,
                'all_accuracies': accuracies
            }
            
            print(f"  Accuracy: {avg_acc:.4f} ± {ci:.4f}")
        
        return results


def run_baseline_for_model(backbone_name, dataset, config):
    """Run baseline evaluation for a specific backbone"""
    
    print(f"\n{'='*80}")
    print(f"LOADING BACKBONE: {backbone_name.upper()}")
    print(f"{'='*80}")
    
    # Create frozen pretrained backbone
    backbone, embedding_dim = BackboneFactory.create_backbone(
        backbone_name=backbone_name,
        pretrained=True,
        freeze=True  # Freeze for baseline
    )
    
    # Wrap with flatten layer
    class FlattenBackbone(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        
        def forward(self, x):
            features = self.backbone(x)
            if len(features.shape) > 2:
                return features.view(features.size(0), -1)
            return features
    
    backbone = FlattenBackbone(backbone)
    
    # Create baseline model
    model = FrozenPrototypicalBaseline(backbone, embedding_dim)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {backbone_name}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"All parameters frozen (no training)")
    
    # Create evaluator
    evaluator = BaselineEvaluator(model, config, dataset, model_name=backbone_name)
    
    # Evaluate
    results = evaluator.evaluate_shots(
        n_way=3,
        k_shots=[1, 5, 10],
        n_query=10,
        n_episodes=100
    )
    
    return results


def print_comparison_table(all_results):
    """Print comparison table of all baselines"""
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON ACROSS BACKBONES")
    print("="*80)
    
    print(f"\n{'Model':<20} {'1-shot':<20} {'5-shot':<20} {'10-shot':<20}")
    print("-"*80)
    
    random_chance = 1/3
    
    for model_name, results in all_results.items():
        print(f"{model_name:<20}", end='')
        for shot in ['1-shot', '5-shot', '10-shot']:
            if shot in results:
                acc = results[shot]['accuracy']
                ci = results[shot]['ci']
                vs_random = acc / random_chance
                print(f"{acc:.4f}±{ci:.4f} ({vs_random:.2f}x)  ", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()
    
    print("-"*80)
    print(f"{'Random Chance':<20} {random_chance:.4f}{'':<15} {random_chance:.4f}{'':<15} {random_chance:.4f}")
    print("="*80)
    
    # Improvement analysis
    print("\n" + "="*80)
    print("IMPROVEMENT WITH MORE SHOTS (Baseline)")
    print("="*80)
    print("\nThis shows how much frozen pretrained features benefit from more examples")
    print("-"*80)
    
    print(f"\n{'Model':<20} {'1→5 shot':<20} {'5→10 shot':<20} {'1→10 shot':<20}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<20}", end='')
        
        # 1 → 5 shot improvement
        if '1-shot' in results and '5-shot' in results:
            acc_1 = results['1-shot']['accuracy']
            acc_5 = results['5-shot']['accuracy']
            imp_1_5 = (acc_5 - acc_1) / acc_1 * 100
            print(f"{imp_1_5:+.1f}%{'':<15}", end='')
        else:
            print(f"{'N/A':<20}", end='')
        
        # 5 → 10 shot improvement
        if '5-shot' in results and '10-shot' in results:
            acc_5 = results['5-shot']['accuracy']
            acc_10 = results['10-shot']['accuracy']
            imp_5_10 = (acc_10 - acc_5) / acc_5 * 100
            print(f"{imp_5_10:+.1f}%{'':<15}", end='')
        else:
            print(f"{'N/A':<20}", end='')
        
        # 1 → 10 shot improvement
        if '1-shot' in results and '10-shot' in results:
            acc_1 = results['1-shot']['accuracy']
            acc_10 = results['10-shot']['accuracy']
            imp_1_10 = (acc_10 - acc_1) / acc_1 * 100
            print(f"{imp_1_10:+.1f}%", end='')
        
        print()
    
    print("="*80)


def save_all_baseline_results(all_results, save_path):
    """Save all baseline results to text file"""
    
    with open(save_path, 'w') as f:
        f.write("BASELINE EVALUATION: ALL BACKBONES\n")
        f.write("="*80 + "\n")
        f.write("Method: Frozen Pretrained Features + Prototypical Networks\n")
        f.write("="*80 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n{model_name.upper()}\n")
            f.write("-"*80 + "\n")
            for shot_name, result in results.items():
                acc = result['accuracy']
                ci = result['ci']
                f.write(f"3-way {shot_name}: {acc:.4f} ± {ci:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Random chance (3-way): 0.3333\n")
        f.write("="*80 + "\n")
    
    print(f"\n[SAVED] All baseline results saved to: {save_path}")


def main():
    """Run baseline evaluation for all models"""
    
    # Configuration
    Config.DATASET_PATH = "data/chexpert"
    Config.setup_directories()
    
    print("\n" + "="*80)
    print("BASELINE EVALUATION FOR ALL BACKBONES")
    print("="*80)
    print("\nThis evaluates frozen pretrained features using Prototypical Networks")
    print("No training - just testing transfer learning capability")
    print("\nBackbones to test:")
    print("  1. ResNet18 (ImageNet pretrained)")
    print("  2. DenseNet121 (ImageNet/Medical pretrained)")
    print("  3. ViT-Base (ImageNet pretrained)")
    print("="*80 + "\n")
    
    # Check dataset
    if not Path(Config.DATASET_PATH).exists():
        print(f"\n[ERROR] Dataset path does not exist: {Config.DATASET_PATH}")
        sys.exit(1)
    
    # Load dataset
    print("Loading CheXpert dataset...\n")
    from chexpert_loader import create_chexpert_dataloaders_fast
    
    try:
        _, _, test_dataset = create_chexpert_dataloaders_fast(
            Config,
            max_images_per_class=500
        )
    except Exception as e:
        print(f"\n[ERROR] loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Number of classes: {len(test_dataset.classes)}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Config.DEVICE = torch.device(device)
    print(f"Using device: {device}\n")
    
    # Models to evaluate
    models_to_test = [
        ('resnet18', 'ResNet18'),
        ('densenet121', 'DenseNet121-Medical'),
        ('vit_base', 'ViT-Base'),
    ]
    
    all_results = {}
    
    # Run evaluation for each model
    for backbone_name, display_name in models_to_test:
        try:
            results = run_baseline_for_model(backbone_name, test_dataset, Config)
            all_results[display_name] = results
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison
    if all_results:
        print_comparison_table(all_results)
        
        # Save results
        save_path = Config.RESULTS_DIR / "all_baseline_results.txt"
        save_all_baseline_results(all_results, save_path)
    
    # Summary
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. Baseline = Frozen pretrained features (no domain adaptation)")
    print("2. Higher accuracy = Better transfer learning from ImageNet")
    print("3. Improvement with shots = Better prototype quality with more examples")
    print("\n4. Next step: Compare with few-shot TRAINED models")
    print("   - Baseline shows what pretrained features can do")
    print("   - Few-shot training adds domain adaptation")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

