"""
Configuration management for Few-Shot Learning framework
"""
import torch
from pathlib import Path


class Config:
    """Configuration class for few-shot learning experiments"""
    
    # Dataset settings
    DATASET_PATH = "path/to/dataset"  # Override this with actual path
    IMAGE_SIZE = 224
    NUM_CHANNELS = 3
    
    # Few-shot learning parameters
    N_WAY = 5  # Number of classes per episode
    K_SHOT = 10  # Number of examples per class in support set
    N_QUERY = 15  # Number of query examples per class
    NUM_EPISODES = 1000  # Number of training episodes
    NUM_EVAL_EPISODES = 600  # Number of evaluation episodes
    
    # Model settings
    BACKBONE = "resnet50"  # Options: resnet50, resnet18, vit_base, densenet121, efficientnet
    EMBEDDING_DIM = 1600  # Embedding dimension (depends on backbone)
    PRETRAINED = True
    FREEZE_BACKBONE = False
    
    # Training settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 100
    SCHEDULER_STEP = 20
    SCHEDULER_GAMMA = 0.5
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.5
    
    # Hardware settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Logging and checkpointing
    USE_WANDB = False  # Set to True to enable Weights & Biases logging
    WANDB_PROJECT = "few-shot-medical-imaging"
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 10
    
    # Evaluation settings
    EVAL_SHOTS = [1, 5, 10, 20]  # Different k-shot scenarios to evaluate
    EVAL_WAYS = [2, 5, 10]  # Different n-way scenarios to evaluate
    
    @classmethod
    def update_embedding_dim(cls):
        """Update embedding dimension based on backbone"""
        backbone_dims = {
            "resnet18": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "densenet121": 1024,
            "vit_base": 768,
            "vit_large": 1024,
            "efficientnet": 1280,
            # OpenCLIP models
            "clip_vit_b32": 512,
            "clip_vit_b16": 512,
            "clip_vit_l14": 768,
            "clip_rn50": 1024,
            "clip_convnext_base": 512,
            "clip_convnext_large": 768,
            # Medical models
            "chexnet": 1024,
            "densenet121_medical": 1024,
        }
        cls.EMBEDDING_DIM = backbone_dims.get(cls.BACKBONE, 2048)
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    # Test configuration
    Config.setup_directories()
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Embedding dimension: {Config.EMBEDDING_DIM}")

