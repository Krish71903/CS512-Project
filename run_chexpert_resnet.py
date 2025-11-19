"""
Fast Few-Shot Learning on CheXpert - Optimized for Speed
Uses smaller dataset and faster model
"""
import sys
import torch
import random
import numpy as np
from pathlib import Path

from config import Config
from chexpert_loader import create_chexpert_dataloaders
from models import create_model
from train import Trainer, AdvancedEvaluator
from visualize import Visualizer


def check_gpu():
    """Check GPU availability and print information"""
    if torch.cuda.is_available():
        print(f"\n✓ GPU is available!")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        return True
    else:
        print("\n✗ WARNING: GPU is NOT available!")
        print("  PyTorch will run on CPU, which will be much slower.")
        print("  Please check:")
        print("    - NVIDIA GPU drivers are installed")
        print("    - CUDA toolkit is installed")
        print("    - PyTorch was installed with CUDA support")
        return False


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Run FAST few-shot learning on CheXpert dataset"""
    
    # ========================================
    # FAST CONFIGURATION
    # ========================================
    Config.DATASET_PATH = "data/chexpert"
    
    # Use smaller, faster model
    Config.BACKBONE = "resnet18"  # Much faster than resnet50!
    Config.PRETRAINED = True
    Config.FREEZE_BACKBONE = False
    
    # Reduce few-shot settings for speed
    Config.N_WAY = 3  # Reduced from 5 to 3 classes
    Config.K_SHOT = 5  # Reduced from 10 to 5 support examples
    Config.N_QUERY = 10  # Reduced from 15 to 10 query examples
    
    # Drastically reduce training time
    Config.NUM_EPOCHS = 20  # Reduced from 50 to 20
    Config.NUM_EPISODES = 100  # Reduced from 500 to 100 per epoch!
    Config.LEARNING_RATE = 1e-3
    Config.USE_AUGMENTATION = True
    
    # Reduce evaluation episodes for speed
    Config.NUM_EVAL_EPISODES = 100  # Reduced from 300 to 100
    Config.EVAL_SHOTS = [1, 5, 10]  # Skip 20-shot for speed
    Config.EVAL_WAYS = [2, 3]  # Only test 2-way and 3-way
    
    # Logging
    Config.USE_WANDB = False
    
    # Setup
    set_seed(42)
    Config.setup_directories()
    Config.update_embedding_dim()
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    print("\n" + "="*80)
    print("FAST FEW-SHOT LEARNING ON CHEXPERT (OPTIMIZED FOR SPEED)")
    print("="*80)
    print(f"\nDataset: {Config.DATASET_PATH}")
    print(f"Backbone: {Config.BACKBONE} (lightweight)")
    print(f"Configuration: {Config.N_WAY}-way {Config.K_SHOT}-shot")
    print(f"Episodes per epoch: {Config.NUM_EPISODES} (10x faster!)")
    print(f"Total epochs: {Config.NUM_EPOCHS}")
    print(f"Device: {Config.DEVICE}")
    if gpu_available:
        print(f"\nEstimated time: ~5-15 minutes on GPU")
    else:
        print(f"\nEstimated time: ~30-60 minutes on CPU")
    print("="*80 + "\n")
    
    # Check if dataset path exists
    if not Path(Config.DATASET_PATH).exists():
        print(f"\n[ERROR] Dataset path does not exist: {Config.DATASET_PATH}")
        sys.exit(1)
    
    # ========================================
    # LOAD DATA (with smaller subset)
    # ========================================
    print("Loading CheXpert dataset (smaller subset for speed)...\n")
    
    # Import the custom fast loader
    from chexpert_loader import create_chexpert_dataloaders_fast
    
    try:
        train_dataset, val_dataset, test_dataset = create_chexpert_dataloaders_fast(
            Config, 
            max_images_per_class=1000  # Limit to 1000 images per class
        )
    except Exception as e:
        print(f"\n[ERROR] loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================
    # CREATE MODEL
    # ========================================
    print("\nCreating Prototypical Network...")
    model = create_model(Config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================
    # TRAINING
    # ========================================
    print("\n" + "="*80)
    print("TRAINING PHASE (Fast Mode)")
    print("="*80 + "\n")
    
    trainer = Trainer(model, Config, train_dataset, val_dataset)
    best_val_acc = trainer.train()
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # ========================================
    # EVALUATION
    # ========================================
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80 + "\n")
    
    # Load best model
    checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}\n")
    
    model.to(Config.DEVICE)
    model.eval()
    
    # Create evaluator
    evaluator = AdvancedEvaluator(model, Config, test_dataset)
    visualizer = Visualizer(Config.RESULTS_DIR)
    
    # Evaluate different k-shot scenarios
    print("\n" + "-"*80)
    print("EVALUATING DIFFERENT K-SHOT SCENARIOS")
    print("-"*80)
    
    shot_results = evaluator.evaluate_multiple_shots(
        n_way=Config.N_WAY,
        k_shots=Config.EVAL_SHOTS,
        n_query=Config.N_QUERY,
        n_episodes=Config.NUM_EVAL_EPISODES
    )
    
    # Evaluate different n-way scenarios
    way_results = None
    if len(test_dataset.classes) >= max(Config.EVAL_WAYS):
        print("\n" + "-"*80)
        print("EVALUATING DIFFERENT N-WAY SCENARIOS")
        print("-"*80)
        
        way_results = evaluator.evaluate_multiple_ways(
            n_ways=Config.EVAL_WAYS,
            k_shot=Config.K_SHOT,
            n_query=Config.N_QUERY,
            n_episodes=Config.NUM_EVAL_EPISODES
        )
    
    # ========================================
    # VISUALIZATION
    # ========================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Training curves
    visualizer.plot_training_curves(
        trainer.train_losses,
        trainer.train_accs,
        trainer.val_accs,
        save_name="training_curves_fast.png"
    )
    
    # Shot comparison
    visualizer.plot_shot_comparison(shot_results, n_way=Config.N_WAY, 
                                    save_name="shot_comparison_fast.png")
    
    # Way comparison
    if way_results:
        visualizer.plot_way_comparison(way_results, k_shot=Config.K_SHOT,
                                       save_name="way_comparison_fast.png")
    
    # Accuracy distributions
    visualizer.plot_accuracy_distribution(shot_results,
                                         save_name="accuracy_distribution_fast.png")
    
    # Comprehensive results
    visualizer.plot_comprehensive_results(shot_results, way_results)
    
    # Results summary
    print("\n" + "-"*80)
    print("K-SHOT EVALUATION SUMMARY")
    print("-"*80)
    visualizer.create_results_summary(shot_results, "chexpert_fast_shot_results.csv")
    
    if way_results:
        print("\n" + "-"*80)
        print("N-WAY EVALUATION SUMMARY")
        print("-"*80)
        visualizer.create_results_summary(way_results, "chexpert_fast_way_results.csv")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("[SUCCESS] FAST EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"\nResults saved to:")
    print(f"   Checkpoints: {Config.CHECKPOINT_DIR}")
    print(f"   Visualizations: {Config.RESULTS_DIR}")
    print("\nGenerated files:")
    print("   - training_curves_fast.png")
    print("   - shot_comparison_fast.png")
    if way_results:
        print("   - way_comparison_fast.png")
    print("   - accuracy_distribution_fast.png")
    print("   - comprehensive_results.png")
    print("   - chexpert_fast_shot_results.csv")
    if way_results:
        print("   - chexpert_fast_way_results.csv")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

