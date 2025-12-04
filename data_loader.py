"""
Generic data loader for few-shot learning
Supports episodic sampling for N-way K-shot learning
"""
import os
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FewShotDataset(Dataset):
    """
    Generic few-shot learning dataset
    Expects data organized as: dataset_path/split/class_name/image_files
    Or: dataset_path/class_name/image_files (will split automatically)
    """
    
    def __init__(self, data_path, split='train', transform=None, min_samples_per_class=20):
        """
        Args:
            data_path: Path to dataset root directory
            split: 'train', 'val', or 'test'
            transform: Augmentation transform function
            min_samples_per_class: Minimum number of samples required per class
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.min_samples_per_class = min_samples_per_class
        
        # Load data
        self.class_to_images = defaultdict(list)
        self.classes = []
        self._load_data()
        
        print(f"\n{split.upper()} Dataset Statistics:")
        print(f"  Total classes: {len(self.classes)}")
        print(f"  Total images: {sum(len(imgs) for imgs in self.class_to_images.values())}")
        print(f"  Images per class: min={min(len(imgs) for imgs in self.class_to_images.values())}, "
              f"max={max(len(imgs) for imgs in self.class_to_images.values())}, "
              f"mean={np.mean([len(imgs) for imgs in self.class_to_images.values()]):.1f}")
    
    def _load_data(self):
        """Load image paths organized by class"""
        # Check if data is organized with split folders
        split_path = self.data_path / self.split
        if split_path.exists():
            data_root = split_path
        else:
            # Data not split, use entire dataset (will warn user)
            data_root = self.data_path
            print(f"Warning: No '{self.split}' folder found. Using entire dataset.")
        
        # Scan for classes (subdirectories)
        if not data_root.exists():
            raise ValueError(f"Dataset path does not exist: {data_root}")
        
        class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {data_root}")
        
        # Load images for each class
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_dir in tqdm(class_dirs, desc=f"Loading {self.split} classes"):
            class_name = class_dir.name
            
            # Find all images in class directory
            images = []
            for ext in valid_extensions:
                images.extend(list(class_dir.glob(f"*{ext}")))
                images.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            # Filter out hidden files and ensure uniqueness
            images = list(set([img for img in images if not img.name.startswith('.')]))
            
            if len(images) >= self.min_samples_per_class:
                self.class_to_images[class_name] = images
                self.classes.append(class_name)
            else:
                print(f"  Skipping class '{class_name}': only {len(images)} images "
                      f"(minimum: {self.min_samples_per_class})")
        
        self.classes = sorted(self.classes)
        
        if len(self.classes) == 0:
            raise ValueError(f"No classes with sufficient samples found in {data_root}")
    
    def __len__(self):
        """Return total number of images"""
        return sum(len(imgs) for imgs in self.class_to_images.values())
    
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image, is_train=(self.split == 'train'))
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                return self.transform(blank, is_train=False)
            return torch.zeros(3, 224, 224)
    
    def sample_episode(self, n_way, k_shot, n_query):
        """
        Sample an episode for few-shot learning
        Args:
            n_way: Number of classes
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
        Returns:
            support_images: [n_way * k_shot, C, H, W]
            support_labels: [n_way * k_shot]
            query_images: [n_way * n_query, C, H, W]
            query_labels: [n_way * n_query]
        """
        # Sample n_way classes
        episode_classes = random.sample(self.classes, n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for label_idx, class_name in enumerate(episode_classes):
            # Get all images for this class
            class_images = self.class_to_images[class_name]
            
            # Sample k_shot + n_query images
            n_samples = k_shot + n_query
            if len(class_images) < n_samples:
                # Sample with replacement if not enough images
                sampled_images = random.choices(class_images, k=n_samples)
            else:
                sampled_images = random.sample(class_images, n_samples)
            
            # Split into support and query
            support_imgs = sampled_images[:k_shot]
            query_imgs = sampled_images[k_shot:]
            
            # Load and transform images
            for img_path in support_imgs:
                img = self.load_image(img_path)
                support_images.append(img)
                support_labels.append(label_idx)
            
            for img_path in query_imgs:
                img = self.load_image(img_path)
                query_images.append(img)
                query_labels.append(label_idx)
        
        # Stack into tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return support_images, support_labels, query_images, query_labels


class EpisodicBatchSampler:
    """Batch sampler that yields episodes for few-shot learning"""
    
    def __init__(self, dataset, n_way, k_shot, n_query, n_episodes):
        """
        Args:
            dataset: FewShotDataset instance
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes per epoch
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.dataset.sample_episode(self.n_way, self.k_shot, self.n_query)
    
    def __len__(self):
        return self.n_episodes


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders
    Args:
        config: Configuration object
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from augmentation import MedicalImageAugmentation
    
    # Initialize augmentation
    aug = MedicalImageAugmentation(
        image_size=config.IMAGE_SIZE,
        strength=config.AUGMENTATION_STRENGTH,
        use_advanced=config.USE_AUGMENTATION
    )
    
    # Create datasets
    train_dataset = FewShotDataset(
        data_path=config.DATASET_PATH,
        split='train',
        transform=aug,
        min_samples_per_class=config.K_SHOT + config.N_QUERY
    )
    
    val_dataset = FewShotDataset(
        data_path=config.DATASET_PATH,
        split='val',
        transform=aug,
        min_samples_per_class=config.K_SHOT + config.N_QUERY
    )
    
    # Try to load test set, if not available use val set
    try:
        test_dataset = FewShotDataset(
            data_path=config.DATASET_PATH,
            split='test',
            transform=aug,
            min_samples_per_class=config.K_SHOT + config.N_QUERY
        )
    except:
        print("No test set found, using validation set for testing")
        test_dataset = val_dataset
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test data loader
    from config import Config
    from augmentation import MedicalImageAugmentation
    
    Config.DATASET_PATH = "path/to/test/dataset"  # Change this for testing
    
    aug = MedicalImageAugmentation(image_size=224)
    dataset = FewShotDataset(
        data_path=Config.DATASET_PATH,
        split='train',
        transform=aug,
        min_samples_per_class=20
    )
    
    # Sample an episode
    support_imgs, support_labels, query_imgs, query_labels = dataset.sample_episode(
        n_way=5, k_shot=5, n_query=10
    )
    
    print(f"Support images shape: {support_imgs.shape}")
    print(f"Support labels shape: {support_labels.shape}")
    print(f"Query images shape: {query_imgs.shape}")
    print(f"Query labels shape: {query_labels.shape}")

