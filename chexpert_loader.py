"""
Fast CheXpert data loader with limited dataset size
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm


class CheXpertFewShotDatasetFast(Dataset):
    """
    Fast CheXpert dataset loader - limits images per class for speed
    """
    
    def __init__(self, data_root, split='train', transform=None, 
                 pathologies=None, min_samples_per_class=20, max_images_per_class=1000):
        """
        Args:
            data_root: Path to CheXpert dataset root
            split: 'train' or 'valid'
            transform: Augmentation transform function
            pathologies: List of pathologies to use
            min_samples_per_class: Minimum samples required
            max_images_per_class: Maximum images to load per class (for speed!)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.min_samples_per_class = min_samples_per_class
        self.max_images_per_class = max_images_per_class
        
        # Select most common pathologies
        if pathologies is None:
            self.pathologies = [
                'No Finding',
                'Cardiomegaly',
                'Edema',
                'Consolidation',
                'Atelectasis',
                'Pleural Effusion'
            ]
        else:
            self.pathologies = pathologies
        
        # Load data
        self.class_to_images = {}
        self.classes = []
        self._load_from_csv()
        
        print(f"\n{split.upper()} CheXpert Dataset (Fast Mode):")
        print(f"  Total classes: {len(self.classes)}")
        print(f"  Total images: {sum(len(imgs) for imgs in self.class_to_images.values())}")
        if self.class_to_images:
            print(f"  Images per class: min={min(len(imgs) for imgs in self.class_to_images.values())}, "
                  f"max={max(len(imgs) for imgs in self.class_to_images.values())}, "
                  f"mean={np.mean([len(imgs) for imgs in self.class_to_images.values()]):.1f}")
    
    def _load_from_csv(self):
        """Load image paths from CSV file with size limit"""
        csv_file = self.data_root / f"{self.split}.csv"
        if not csv_file.exists():
            raise ValueError(f"CSV file not found: {csv_file}")
        
        print(f"Loading CheXpert from {csv_file} (limited to {self.max_images_per_class} per class)...")
        df = pd.read_csv(csv_file)
        
        for pathology in tqdm(self.pathologies, desc="Processing pathologies"):
            if pathology not in df.columns:
                continue
            
            # Get positive samples
            positive_mask = df[pathology] == 1.0
            positive_df = df[positive_mask]
            
            if len(positive_df) >= self.min_samples_per_class:
                # Randomly sample subset if too many images
                if len(positive_df) > self.max_images_per_class:
                    positive_df = positive_df.sample(n=self.max_images_per_class, random_state=42)
                
                # Load image paths
                image_paths = []
                for _, row in positive_df.iterrows():
                    csv_path = row['Path']
                    
                    # Remove CheXpert prefix
                    if 'CheXpert-v1.0-small/' in csv_path:
                        csv_path = csv_path.replace('CheXpert-v1.0-small/', '')
                    elif 'CheXpert-v1.0/' in csv_path:
                        csv_path = csv_path.replace('CheXpert-v1.0/', '')
                    
                    img_path = self.data_root / csv_path
                    if img_path.exists():
                        image_paths.append(img_path)
                    
                    # Stop early if we have enough
                    if len(image_paths) >= self.max_images_per_class:
                        break
                
                if len(image_paths) >= self.min_samples_per_class:
                    self.class_to_images[pathology] = image_paths
                    self.classes.append(pathology)
                    print(f"  [OK] {pathology}: {len(image_paths)} images")
        
        if len(self.classes) == 0:
            raise ValueError("No classes with sufficient samples found!")
        
        self.classes = sorted(self.classes)
    
    def __len__(self):
        return sum(len(imgs) for imgs in self.class_to_images.values())
    
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image, is_train=(self.split == 'train'))
            return image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                return self.transform(blank, is_train=False)
            return torch.zeros(3, 224, 224)
    
    def sample_episode(self, n_way, k_shot, n_query):
        """Sample an episode for few-shot learning"""
        episode_classes = random.sample(self.classes, n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for label_idx, class_name in enumerate(episode_classes):
            class_images = self.class_to_images[class_name]
            
            n_samples = k_shot + n_query
            if len(class_images) < n_samples:
                sampled_images = random.choices(class_images, k=n_samples)
            else:
                sampled_images = random.sample(class_images, n_samples)
            
            support_imgs = sampled_images[:k_shot]
            query_imgs = sampled_images[k_shot:]
            
            for img_path in support_imgs:
                img = self.load_image(img_path)
                support_images.append(img)
                support_labels.append(label_idx)
            
            for img_path in query_imgs:
                img = self.load_image(img_path)
                query_images.append(img)
                query_labels.append(label_idx)
        
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return support_images, support_labels, query_images, query_labels


def create_chexpert_dataloaders_fast(config, max_images_per_class=1000):
    """
    Create fast CheXpert dataloaders with limited dataset size
    """
    from augmentation import MedicalImageAugmentation
    
    aug = MedicalImageAugmentation(
        image_size=config.IMAGE_SIZE,
        strength=config.AUGMENTATION_STRENGTH,
        use_advanced=config.USE_AUGMENTATION
    )
    
    train_dataset = CheXpertFewShotDatasetFast(
        data_root=config.DATASET_PATH,
        split='train',
        transform=aug,
        min_samples_per_class=config.K_SHOT + config.N_QUERY,
        max_images_per_class=max_images_per_class
    )
    
    val_dataset = CheXpertFewShotDatasetFast(
        data_root=config.DATASET_PATH,
        split='valid',
        transform=aug,
        min_samples_per_class=config.K_SHOT + config.N_QUERY,
        max_images_per_class=max_images_per_class // 2  # Even smaller for validation
    )
    
    test_dataset = val_dataset
    
    return train_dataset, val_dataset, test_dataset

