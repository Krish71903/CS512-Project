"""
Direct CheXpert data loader - works with original CSV format
No need to reorganize the dataset!
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm


class CheXpertFewShotDataset(Dataset):
    """
    CheXpert dataset loader for few-shot learning
    Works directly with the original CSV files - no reorganization needed!
    """
    
    def __init__(self, data_root, split='train', transform=None, 
                 pathologies=None, min_samples_per_class=20):
        """
        Args:
            data_root: Path to CheXpert dataset root (e.g., 'data/chexpert')
            split: 'train' or 'valid'
            transform: Augmentation transform function
            pathologies: List of pathologies to use (None = use default)
            min_samples_per_class: Minimum samples required per class
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.min_samples_per_class = min_samples_per_class
        
        # Default pathologies (most common ones with enough samples)
        if pathologies is None:
            self.pathologies = [
                'No Finding',
                'Cardiomegaly',
                'Edema',
                'Consolidation',
                'Atelectasis',
                'Pleural Effusion',
                'Pneumonia',
                'Pneumothorax'
            ]
        else:
            self.pathologies = pathologies
        
        # Load data
        self.class_to_images = {}
        self.classes = []
        self._load_from_csv()
        
        print(f"\n{split.upper()} CheXpert Dataset Statistics:")
        print(f"  Total classes: {len(self.classes)}")
        print(f"  Total images: {sum(len(imgs) for imgs in self.class_to_images.values())}")
        if self.class_to_images:
            print(f"  Images per class: min={min(len(imgs) for imgs in self.class_to_images.values())}, "
                  f"max={max(len(imgs) for imgs in self.class_to_images.values())}, "
                  f"mean={np.mean([len(imgs) for imgs in self.class_to_images.values()]):.1f}")
    
    def _load_from_csv(self):
        """Load image paths and labels from CSV file"""
        # Determine CSV file path
        csv_file = self.data_root / f"{self.split}.csv"
        if not csv_file.exists():
            raise ValueError(f"CSV file not found: {csv_file}")
        
        print(f"Loading CheXpert from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"Total images in CSV: {len(df)}")
        
        # Process each pathology
        for pathology in tqdm(self.pathologies, desc="Processing pathologies"):
            if pathology not in df.columns:
                print(f"  Warning: '{pathology}' not found in CSV, skipping...")
                continue
            
            # Get images with positive labels (1.0 = positive)
            # In CheXpert: 1.0 = positive, 0.0 = negative, -1.0 = uncertain, NaN = not mentioned
            positive_mask = df[pathology] == 1.0
            positive_df = df[positive_mask]
            
            if len(positive_df) >= self.min_samples_per_class:
                # Store image paths for this class
                image_paths = []
                for _, row in positive_df.iterrows():
                    # CheXpert CSV paths include "CheXpert-v1.0-small/" prefix
                    # We need to strip that and use actual data_root
                    csv_path = row['Path']
                    
                    # Remove CheXpert prefix if present
                    if 'CheXpert-v1.0-small/' in csv_path:
                        csv_path = csv_path.replace('CheXpert-v1.0-small/', '')
                    elif 'CheXpert-v1.0/' in csv_path:
                        csv_path = csv_path.replace('CheXpert-v1.0/', '')
                    
                    img_path = self.data_root / csv_path
                    if img_path.exists():
                        image_paths.append(img_path)
                
                if len(image_paths) >= self.min_samples_per_class:
                    self.class_to_images[pathology] = image_paths
                    self.classes.append(pathology)
                    print(f"  [OK] {pathology}: {len(image_paths)} images")
                else:
                    print(f"  [SKIP] {pathology}: only {len(image_paths)} valid images "
                          f"(minimum: {self.min_samples_per_class})")
            else:
                print(f"  [SKIP] {pathology}: only {len(positive_df)} images "
                      f"(minimum: {self.min_samples_per_class})")
        
        if len(self.classes) == 0:
            raise ValueError("No classes with sufficient samples found!")
        
        self.classes = sorted(self.classes)
    
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


def create_chexpert_dataloaders(config):
    """
    Create CheXpert train and validation dataloaders
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
    train_dataset = CheXpertFewShotDataset(
        data_root=config.DATASET_PATH,
        split='train',
        transform=aug,
        min_samples_per_class=config.K_SHOT + config.N_QUERY
    )
    
    val_dataset = CheXpertFewShotDataset(
        data_root=config.DATASET_PATH,
        split='valid',
        transform=aug,
        min_samples_per_class=config.K_SHOT + config.N_QUERY
    )
    
    # Use validation set for testing (CheXpert doesn't have separate test set)
    test_dataset = val_dataset
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the loader
    from config import Config
    from augmentation import MedicalImageAugmentation
    
    Config.DATASET_PATH = "data/chexpert"
    Config.K_SHOT = 5
    Config.N_QUERY = 10
    
    aug = MedicalImageAugmentation(image_size=224)
    
    # Test train dataset
    train_dataset = CheXpertFewShotDataset(
        data_root=Config.DATASET_PATH,
        split='train',
        transform=aug,
        min_samples_per_class=15
    )
    
    # Sample an episode
    support_imgs, support_labels, query_imgs, query_labels = train_dataset.sample_episode(
        n_way=5, k_shot=5, n_query=10
    )
    
    print(f"\nEpisode sampling test:")
    print(f"Support images shape: {support_imgs.shape}")
    print(f"Support labels shape: {support_labels.shape}")
    print(f"Query images shape: {query_imgs.shape}")
    print(f"Query labels shape: {query_labels.shape}")

