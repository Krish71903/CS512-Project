"""
Data augmentation strategies for medical imaging
Implements geometric transformations, intensity adjustments, and advanced augmentations
"""
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


class MedicalImageAugmentation:
    """Comprehensive augmentation pipeline for medical images"""
    
    def __init__(self, image_size=224, strength=0.5, use_advanced=True):
        """
        Args:
            image_size: Target image size
            strength: Augmentation strength (0.0 to 1.0)
            use_advanced: Whether to use advanced augmentations
        """
        self.image_size = image_size
        self.strength = strength
        self.use_advanced = use_advanced
        
        # Basic normalization (ImageNet statistics work well for transfer learning)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def get_train_transform(self):
        """Get training augmentation pipeline"""
        if self.use_advanced:
            return A.Compose([
                # Geometric transformations
                A.Resize(self.image_size, self.image_size),
                A.Rotate(limit=int(15 * self.strength), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05 * self.strength,
                    scale_limit=0.1 * self.strength,
                    rotate_limit=15 * self.strength,
                    p=0.5
                ),
                
                # Intensity adjustments
                A.RandomBrightnessContrast(
                    brightness_limit=0.2 * self.strength,
                    contrast_limit=0.2 * self.strength,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                
                # Advanced augmentations
                A.GaussNoise(var_limit=(10.0, 50.0 * self.strength), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.CLAHE(clip_limit=4.0, p=0.3),
                
                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def get_val_transform(self):
        """Get validation/test augmentation pipeline (no augmentation)"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __call__(self, image, is_train=True):
        """
        Apply augmentation to image
        Args:
            image: PIL Image or numpy array
            is_train: Whether to apply training augmentations
        Returns:
            Augmented image tensor
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        
        transform = self.get_train_transform() if is_train else self.get_val_transform()
        augmented = transform(image=image)
        return augmented['image']


class MixUpAugmentation:
    """MixUp augmentation for improved generalization"""
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """
        Apply MixUp augmentation
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
        Returns:
            Mixed images and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


if __name__ == "__main__":
    # Test augmentation pipeline
    import matplotlib.pyplot as plt
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Initialize augmentation
    aug = MedicalImageAugmentation(image_size=224, strength=0.5)
    
    # Apply augmentation
    augmented = aug(dummy_image, is_train=True)
    print(f"Augmented image shape: {augmented.shape}")
    print(f"Augmented image dtype: {augmented.dtype}")
    print(f"Augmented image range: [{augmented.min():.3f}, {augmented.max():.3f}]")

