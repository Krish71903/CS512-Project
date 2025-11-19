"""
Few-Shot Learning Models
Implements Prototypical Networks with multiple backbone architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from transformers import ViTModel, ViTConfig

# OpenCLIP import (will be used if available)
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("[WARNING] open_clip not installed. Install with: pip install open-clip-torch")


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning
    Reference: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    """
    
    def __init__(self, backbone, embedding_dim=1600):
        """
        Args:
            backbone: Feature extractor backbone network
            embedding_dim: Dimension of embedding space
        """
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
    
    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        """
        Forward pass for Prototypical Networks
        Args:
            support_images: [n_way * k_shot, C, H, W]
            support_labels: [n_way * k_shot]
            query_images: [n_query, C, H, W]
            n_way: Number of classes
            k_shot: Number of support examples per class
        Returns:
            logits: [n_query, n_way]
        """
        # Extract features
        support_features = self.backbone(support_images)  # [n_way * k_shot, embedding_dim]
        query_features = self.backbone(query_images)  # [n_query, embedding_dim]
        
        # Compute prototypes (mean of support features per class)
        prototypes = self._compute_prototypes(support_features, support_labels, n_way)
        
        # Compute distances and convert to logits
        logits = self._compute_logits(query_features, prototypes)
        
        return logits
    
    def _compute_prototypes(self, support_features, support_labels, n_way):
        """
        Compute prototype for each class
        Args:
            support_features: [n_way * k_shot, embedding_dim]
            support_labels: [n_way * k_shot]
            n_way: Number of classes
        Returns:
            prototypes: [n_way, embedding_dim]
        """
        prototypes = []
        for label in range(n_way):
            # Get features for this class
            class_mask = (support_labels == label)
            class_features = support_features[class_mask]
            # Compute mean (prototype)
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # [n_way, embedding_dim]
        return prototypes
    
    def _compute_logits(self, query_features, prototypes):
        """
        Compute logits based on negative Euclidean distance
        Args:
            query_features: [n_query, embedding_dim]
            prototypes: [n_way, embedding_dim]
        Returns:
            logits: [n_query, n_way]
        """
        # Compute Euclidean distances
        n_query = query_features.size(0)
        n_way = prototypes.size(0)
        
        # Expand dimensions for broadcasting
        query_features = query_features.unsqueeze(1)  # [n_query, 1, embedding_dim]
        prototypes = prototypes.unsqueeze(0)  # [1, n_way, embedding_dim]
        
        # Compute squared Euclidean distance
        distances = torch.sum((query_features - prototypes) ** 2, dim=2)  # [n_query, n_way]
        
        # Convert to logits (negative distance)
        logits = -distances
        
        return logits


class BackboneFactory:
    """Factory class for creating different backbone architectures"""
    
    @staticmethod
    def create_backbone(backbone_name, pretrained=True, freeze=False):
        """
        Create a backbone network
        Args:
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze backbone weights
        Returns:
            backbone: Backbone network
            embedding_dim: Output embedding dimension
        """
        backbone_name = backbone_name.lower()
        
        if backbone_name == "resnet18":
            backbone, embedding_dim = BackboneFactory._create_resnet18(pretrained)
        elif backbone_name == "resnet50":
            backbone, embedding_dim = BackboneFactory._create_resnet50(pretrained)
        elif backbone_name == "resnet101":
            backbone, embedding_dim = BackboneFactory._create_resnet101(pretrained)
        elif backbone_name == "densenet121":
            backbone, embedding_dim = BackboneFactory._create_densenet121(pretrained)
        elif backbone_name == "vit_base":
            backbone, embedding_dim = BackboneFactory._create_vit_base(pretrained)
        elif backbone_name == "vit_large":
            backbone, embedding_dim = BackboneFactory._create_vit_large(pretrained)
        elif backbone_name == "efficientnet":
            backbone, embedding_dim = BackboneFactory._create_efficientnet(pretrained)
        elif backbone_name.startswith("clip_"):
            # OpenCLIP models: clip_vit_b32, clip_vit_b16, clip_vit_l14, clip_rn50
            backbone, embedding_dim = BackboneFactory._create_openclip(backbone_name, pretrained)
        elif backbone_name in ["chexnet", "densenet121_medical"]:
            # Medical pretrained models
            backbone, embedding_dim = BackboneFactory._create_medical_densenet121(pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Freeze backbone if requested
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        
        return backbone, embedding_dim
    
    @staticmethod
    def _create_resnet18(pretrained):
        """Create ResNet-18 backbone"""
        resnet = models.resnet18(pretrained=pretrained)
        # Remove final classification layer
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        embedding_dim = 512
        return backbone, embedding_dim
    
    @staticmethod
    def _create_resnet50(pretrained):
        """Create ResNet-50 backbone"""
        resnet = models.resnet50(pretrained=pretrained)
        # Remove final classification layer
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        embedding_dim = 2048
        return backbone, embedding_dim
    
    @staticmethod
    def _create_resnet101(pretrained):
        """Create ResNet-101 backbone"""
        resnet = models.resnet101(pretrained=pretrained)
        # Remove final classification layer
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        embedding_dim = 2048
        return backbone, embedding_dim
    
    @staticmethod
    def _create_densenet121(pretrained):
        """Create DenseNet-121 backbone"""
        densenet = models.densenet121(pretrained=pretrained)
        # Remove final classification layer
        backbone = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        embedding_dim = 1024
        return backbone, embedding_dim
    
    @staticmethod
    def _create_vit_base(pretrained):
        """Create Vision Transformer Base backbone"""
        if pretrained:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        # Remove classification head
        backbone = nn.Sequential(*list(model.children())[:-1])
        embedding_dim = 768
        
        # Wrap to handle ViT output
        class ViTBackbone(nn.Module):
            def __init__(self, vit_model, embed_dim):
                super().__init__()
                self.vit = vit_model
                self.embed_dim = embed_dim
            
            def forward(self, x):
                features = self.vit.forward_features(x)
                # Global average pooling over patch tokens
                return features.mean(dim=1)
        
        backbone = ViTBackbone(model, embedding_dim)
        return backbone, embedding_dim
    
    @staticmethod
    def _create_vit_large(pretrained):
        """Create Vision Transformer Large backbone"""
        if pretrained:
            model = timm.create_model('vit_large_patch16_224', pretrained=True)
        else:
            model = timm.create_model('vit_large_patch16_224', pretrained=False)
        
        embedding_dim = 1024
        
        class ViTBackbone(nn.Module):
            def __init__(self, vit_model, embed_dim):
                super().__init__()
                self.vit = vit_model
                self.embed_dim = embed_dim
            
            def forward(self, x):
                features = self.vit.forward_features(x)
                return features.mean(dim=1)
        
        backbone = ViTBackbone(model, embedding_dim)
        return backbone, embedding_dim
    
    @staticmethod
    def _create_efficientnet(pretrained):
        """Create EfficientNet-B0 backbone"""
        if pretrained:
            efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        else:
            efficientnet = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Remove classification head
        backbone = nn.Sequential(
            efficientnet.conv_stem,
            efficientnet.bn1,
            efficientnet.act1,
            efficientnet.blocks,
            efficientnet.conv_head,
            efficientnet.bn2,
            efficientnet.act2,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        embedding_dim = 1280
        return backbone, embedding_dim
    
    @staticmethod
    def _create_openclip(backbone_name, pretrained):
        """
        Create OpenCLIP backbone
        Supported models:
        - clip_vit_b32: ViT-B/32 (512-dim embeddings)
        - clip_vit_b16: ViT-B/16 (512-dim embeddings)
        - clip_vit_l14: ViT-L/14 (768-dim embeddings)
        - clip_rn50: ResNet-50 (1024-dim embeddings)
        """
        if not OPENCLIP_AVAILABLE:
            raise ImportError(
                "open_clip is not installed. "
                "Install it with: pip install open-clip-torch"
            )
        
        # Map backbone names to OpenCLIP model names
        clip_model_map = {
            'clip_vit_b32': ('ViT-B-32', 512, 'openai'),
            'clip_vit_b16': ('ViT-B-16', 512, 'openai'),
            'clip_vit_l14': ('ViT-L-14', 768, 'openai'),
            'clip_rn50': ('RN50', 1024, 'openai'),
            'clip_convnext_base': ('convnext_base', 512, 'laion400m_s13b_b51k'),
            'clip_convnext_large': ('convnext_large_d', 768, 'laion2b_s26b_b102k_augreg'),
        }
        
        if backbone_name not in clip_model_map:
            available = ', '.join(clip_model_map.keys())
            raise ValueError(
                f"Unknown CLIP model: {backbone_name}. "
                f"Available models: {available}"
            )
        
        model_name, embedding_dim, pretrain_source = clip_model_map[backbone_name]
        
        # Load OpenCLIP model
        if pretrained:
            print(f"Loading OpenCLIP model: {model_name} (pretrained on {pretrain_source})")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrain_source
            )
        else:
            print(f"Loading OpenCLIP model: {model_name} (no pretraining)")
            model = open_clip.create_model(model_name)
        
        # Wrapper to extract visual features
        class CLIPVisualBackbone(nn.Module):
            def __init__(self, clip_model, embed_dim):
                super().__init__()
                self.clip_model = clip_model
                self.embed_dim = embed_dim
            
            def forward(self, x):
                # Extract visual features from CLIP
                # Normalize input to CLIP's expected range if needed
                features = self.clip_model.encode_image(x)
                # Features are already normalized in CLIP
                return features.float()
        
        backbone = CLIPVisualBackbone(model, embedding_dim)
        
        print(f"OpenCLIP backbone created: {model_name}")
        print(f"Embedding dimension: {embedding_dim}")
        
        return backbone, embedding_dim
    
    @staticmethod
    def _create_medical_densenet121(pretrained):
        """
        Create DenseNet-121 for medical imaging (CheXNet-style)
        Optimized for chest X-ray analysis
        
        Note: This uses ImageNet pretrained weights as base.
        For true CheXNet weights, download from: https://github.com/arnoweng/CheXNet
        
        Returns:
            backbone: Feature extractor
            embedding_dim: Output dimension
        """
        print("Loading Medical DenseNet-121 (CheXNet-style for chest X-rays)...")
        
        densenet = models.densenet121(pretrained=pretrained)
        
        # Remove classification head (same architecture as CheXNet)
        backbone = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        embedding_dim = 1024
        
        if pretrained:
            print("  Using ImageNet pretrained weights as base")
            print("  This architecture matches CheXNet (Rajpurkar et al., 2017)")
        
        return backbone, embedding_dim


class FlattenLayer(nn.Module):
    """Helper layer to flatten feature maps"""
    def forward(self, x):
        return x.view(x.size(0), -1)


def create_model(config):
    """
    Create a Prototypical Network model
    Args:
        config: Configuration object
    Returns:
        model: PrototypicalNetwork instance
    """
    # Create backbone
    backbone, embedding_dim = BackboneFactory.create_backbone(
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        freeze=config.FREEZE_BACKBONE
    )
    
    # Add flatten layer if needed
    backbone = nn.Sequential(
        backbone,
        FlattenLayer()
    )
    
    # Create Prototypical Network
    model = PrototypicalNetwork(backbone, embedding_dim)
    
    # Update config with actual embedding dimension
    config.EMBEDDING_DIM = embedding_dim
    
    return model


if __name__ == "__main__":
    # Test model creation
    from config import Config
    
    print("Testing model creation...")
    
    # Test different backbones
    backbones = ["resnet18", "resnet50", "densenet121", "vit_base"]
    
    for backbone_name in backbones:
        print(f"\nTesting {backbone_name}:")
        Config.BACKBONE = backbone_name
        model = create_model(Config)
        
        # Test forward pass
        support_images = torch.randn(25, 3, 224, 224)  # 5-way 5-shot
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        query_images = torch.randn(15, 3, 224, 224)  # 5 classes, 3 queries each
        
        model.eval()
        with torch.no_grad():
            logits = model(support_images, support_labels, query_images, n_way=5, k_shot=5)
        
        print(f"  Embedding dim: {Config.EMBEDDING_DIM}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Success!")

