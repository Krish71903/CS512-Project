import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split


# ============================================================================
# CHEXPERT DATA LOADER (FOR 4-WAY CLASSIFICATION)
# ============================================================================

class CheXpertDataProcessor:
    """Process CheXpert CSV data for few-shot learning"""

    def __init__(self, csv_path, data_root, uncertain_policy='ignore', frontal_only=True):

        self.csv_path = csv_path
        self.data_root = data_root
        self.uncertain_policy = uncertain_policy

        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV: {len(self.df)} rows")

        # Filter to frontal views only
        if frontal_only:
            before = len(self.df)
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'].copy()
            print(f"✓ Kept only Frontal views: {len(self.df)} / {before}")

        # Disease columns
        self.disease_columns = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]

        print(f"✓ Disease columns: {len(self.disease_columns)}")
        print(f"✓ Uncertain policy: {uncertain_policy}")

    def handle_uncertain(self, value):
        """Handle uncertain (-1) labels and NaNs"""
        if pd.isna(value):
            return None
        if value == -1.0:
            if self.uncertain_policy == 'positive':
                return 1.0
            elif self.uncertain_policy == 'negative':
                return 0.0
            else:  # 'ignore'
                return None
        return value

    def _get_label_series(self, colname):
        """Return a cleaned label series (0/1/None) for a disease column"""
        return self.df[colname].apply(self.handle_uncertain)

    def get_multiclass_dataset_exclusive(self, diseases, min_samples_per_class=100):

        image_paths = []
        labels = []
        class_names = {}

        print(f"\n{'=' * 70}")
        print(f"Creating 4-Way EXCLUSIVE Classification Dataset")
        print(f"{'=' * 70}")

        # Get label series for all diseases
        label_series = {disease: self._get_label_series(disease) for disease in diseases}

        for class_idx, disease in enumerate(diseases):
            class_names[class_idx] = disease

            # Get samples where THIS disease is positive
            disease_mask = (label_series[disease] == 1.0)

            # Ensure OTHER diseases are NOT positive (exclusive)
            for other_disease in diseases:
                if other_disease != disease:
                    disease_mask &= (label_series[other_disease] != 1.0)

            disease_samples = self.df[disease_mask]

            print(f"\n✓ Class {class_idx} ({disease}): {len(disease_samples)} exclusive samples")

            added = 0
            for _, row in disease_samples.iterrows():
                img_path = os.path.join(self.data_root, row['Path'])
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(class_idx)
                    added += 1

            print(f"  → Added {added} valid images")

        # Count per class
        print(f"\n{'=' * 70}")
        print(f"Final Class Distribution:")
        print(f"{'=' * 70}")

        for class_idx in range(len(diseases)):
            count = sum(1 for l in labels if l == class_idx)
            print(f"  Class {class_idx} ({class_names[class_idx]}): {count} images")
            if count < min_samples_per_class:
                print(f"    ⚠ Warning: Only {count} samples (< {min_samples_per_class})")

        print(f"\n✓ Total dataset: {len(image_paths)} images")

        return image_paths, labels, class_names


# ============================================================================
# FEW-SHOT DATASET
# ============================================================================

class FewShotDataset(Dataset):
    """Dataset for episodic few-shot learning with validation"""

    def __init__(self, image_paths, labels, transform=None, min_samples_per_class=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Organize by class
        self.classes = sorted(list(set(labels)))
        self.class_to_indices = {c: [] for c in self.classes}
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        print(f"\n✓ FewShotDataset: {len(self.classes)} classes, {len(image_paths)} images")

        # Validate class sizes
        insufficient_classes = []
        for c in self.classes:
            count = len(self.class_to_indices[c])
            print(f"  Class {c}: {count} images")

            if min_samples_per_class and count < min_samples_per_class:
                insufficient_classes.append(c)
                print(f"    ⚠ WARNING: Only {count} images (< {min_samples_per_class} required)")

        if insufficient_classes:
            print(f"\n⚠ WARNING: {len(insufficient_classes)} classes have insufficient samples!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def sample_episode(self, n_way, n_support, n_query):
        """Sample an N-way K-shot episode"""
        if len(self.classes) < n_way:
            raise ValueError(f"Not enough classes! Have {len(self.classes)}, need {n_way}")

        # Filter classes that have enough samples
        valid_classes = [
            c for c in self.classes
            if len(self.class_to_indices[c]) >= (n_support + n_query)
        ]

        if len(valid_classes) < n_way:
            raise ValueError(
                f"Not enough valid classes! Need {n_way} classes with at least "
                f"{n_support + n_query} samples each. Have {len(valid_classes)} valid classes."
            )

        # Select all classes (for 4-way, we use all 4)
        selected_classes = random.sample(valid_classes, n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for new_label, cls in enumerate(selected_classes):
            indices = random.sample(self.class_to_indices[cls], n_support + n_query)

            for i, idx in enumerate(indices):
                image = Image.open(self.image_paths[idx]).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                if i < n_support:
                    support_images.append(image)
                    support_labels.append(new_label)
                else:
                    query_images.append(image)
                    query_labels.append(new_label)

        return (
            torch.stack(support_images),
            torch.tensor(support_labels),
            torch.stack(query_images),
            torch.tensor(query_labels)
        )


# ============================================================================
# PROTOTYPICAL NETWORK
# ============================================================================

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, embedding_dim=768, projection_dim=256,
                 freeze_encoder=True):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder frozen for few-shot learning")
        else:
            print("✓ Encoder remains trainable")

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        outputs = self.encoder(x)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]

        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


# ============================================================================
# PROTOTYPICAL LOSS AND TRAINING
# ============================================================================

def prototypical_loss(embeddings, labels, n_support):
    """Compute prototypical loss"""
    classes = torch.unique(labels)
    n_classes = len(classes)
    n_query = embeddings.shape[0] // n_classes - n_support

    support_idx = torch.stack([torch.where(labels == c)[0][:n_support] for c in classes]).flatten()
    query_idx = torch.stack([torch.where(labels == c)[0][n_support:] for c in classes]).flatten()

    support_embeddings = embeddings[support_idx]
    query_embeddings = embeddings[query_idx]

    support_embeddings = support_embeddings.view(n_classes, n_support, -1)
    prototypes = support_embeddings.mean(dim=1)

    distances = torch.cdist(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-distances, dim=1)

    query_labels = labels[query_idx]
    label_to_idx = {c.item(): i for i, c in enumerate(classes)}
    query_label_indices = torch.tensor([label_to_idx[l.item()] for l in query_labels],
                                       device=embeddings.device)

    loss = F.nll_loss(log_p_y, query_label_indices)
    predictions = torch.argmin(distances, dim=1)
    accuracy = (predictions == query_label_indices).float().mean()

    return loss, accuracy


def train_episode(model, dataset, n_way, n_support, n_query, optimizer, device):
    """Train one episode"""
    model.train()

    support_imgs, support_labels, query_imgs, query_labels = dataset.sample_episode(
        n_way, n_support, n_query
    )

    support_imgs = support_imgs.to(device)
    support_labels = support_labels.to(device)
    query_imgs = query_imgs.to(device)
    query_labels = query_labels.to(device)

    all_images = torch.cat([support_imgs, query_imgs], dim=0)
    all_labels = torch.cat([support_labels, query_labels], dim=0)

    embeddings = model(all_images)
    loss, accuracy = prototypical_loss(embeddings, all_labels, n_support)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), accuracy.item()


def validate_episode(model, dataset, n_way, n_support, n_query, device, n_episodes=100):
    """Validate over multiple episodes"""
    model.eval()

    total_loss = 0
    total_acc = 0

    for _ in tqdm(range(n_episodes), desc='Validation'):
        support_imgs, support_labels, query_imgs, query_labels = dataset.sample_episode(
            n_way, n_support, n_query
        )

        support_imgs = support_imgs.to(device)
        support_labels = support_labels.to(device)
        query_imgs = query_imgs.to(device)
        query_labels = query_labels.to(device)

        all_images = torch.cat([support_imgs, query_imgs], dim=0)
        all_labels = torch.cat([support_labels, query_labels], dim=0)

        with torch.no_grad():
            embeddings = model(all_images)
            loss, accuracy = prototypical_loss(embeddings, all_labels, n_support)

        total_loss += loss.item()
        total_acc += accuracy.item()

    return total_loss / n_episodes, total_acc / n_episodes


# ============================================================================
# MAIN TRAINING - 4-WAY CLASSIFICATION WITH FIXED ENCODER LOADING
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("4-Way Few-Shot Learning with CheXpert (20K Dataset)")
    print("=" * 70)
    print(f"Device: {device}\n")

    # ========================================================================
    # Configuration
    # ========================================================================

    csv_path = 'CheXpert-v1.0-small/train.csv'
    data_root = '/data2/rohit_gene_prediction/chest_testing'

    print(f"✓ CSV path: {csv_path}")
    print(f"✓ Data root: {data_root}\n")

    # ========================================================================
    # Load CheXpert Data
    # ========================================================================

    processor = CheXpertDataProcessor(
        csv_path=csv_path,
        data_root=data_root,
        uncertain_policy='ignore',
        frontal_only=True
    )

    # ========================================================================
    # 4-WAY CLASSIFICATION: Choose 4 Diseases
    # ========================================================================

    # RECOMMENDED: Diseases with good representation
    diseases = [
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Pleural Effusion'
    ]

    print(f"\n{'=' * 70}")
    print(f"Selected Diseases for 4-Way Classification:")
    print(f"{'=' * 70}")
    for i, disease in enumerate(diseases):
        print(f"  Class {i}: {disease}")
    print(f"{'=' * 70}\n")

    # ========================================================================
    # Create Dataset
    # ========================================================================

    image_paths, labels, class_names = processor.get_multiclass_dataset_exclusive(
        diseases=diseases,
        min_samples_per_class=200
    )

    n_way = 4

    if len(image_paths) < 800:
        print("\n❌ ERROR: Insufficient data!")
        print(f"   Got {len(image_paths)} images, need at least 800 (200 per class)")
        return

    # ========================================================================
    # Train/Val Split
    # ========================================================================

    print(f"\n{'=' * 70}")
    print("Splitting Data (80/20 Train/Val)")
    print(f"{'=' * 70}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(f"\nTrain set: {len(train_paths)} images")
    for cls in range(n_way):
        count = sum(1 for l in train_labels if l == cls)
        print(f"  Class {cls} ({class_names[cls]}): {count} images")

    print(f"\nVal set: {len(val_paths)} images")
    for cls in range(n_way):
        count = sum(1 for l in val_labels if l == cls)
        print(f"  Class {cls} ({class_names[cls]}): {count} images")

    # ========================================================================
    # Create Transforms
    # ========================================================================

    image_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ])

    # ========================================================================
    # Few-shot Configuration
    # ========================================================================

    n_support = 10
    n_query = 20
    min_required = n_support + n_query

    print(f"\n{'=' * 70}")
    print(f"Few-Shot Configuration (Optimized for 20K)")
    print(f"{'=' * 70}")
    print(f"  N-way: {n_way}")
    print(f"  K-shot: {n_support}")
    print(f"  Query per class: {n_query}")
    print(f"  Min samples needed: {min_required} per class")
    print(f"{'=' * 70}\n")

    # ========================================================================
    # Create Datasets
    # ========================================================================

    train_dataset = FewShotDataset(
        train_paths, train_labels, train_transform,
        min_samples_per_class=min_required
    )

    val_dataset = FewShotDataset(
        val_paths, val_labels, val_transform,
        min_samples_per_class=min_required
    )

    # ========================================================================
    # Load Model - USE YOUR FINE-TUNED ENCODER
    # ========================================================================

    print(f"\n{'=' * 70}")
    print("Loading Model")
    print(f"{'=' * 70}")

    # Load base RAD-DINO architecture
    encoder = AutoModel.from_pretrained("microsoft/rad-dino")
    encoder_loaded = False
    encoder_source = "base RAD-DINO (not fine-tuned)"

    def load_encoder_state(encoder, state_dict, prefix_to_strip=None, label=""):
        """Helper to load encoder weights with optional prefix stripping"""
        # Strip prefix if specified
        if prefix_to_strip is not None:
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith(prefix_to_strip):
                    new_sd[k[len(prefix_to_strip):]] = v
            state_dict = new_sd

        # Load with strict=False
        load_result = encoder.load_state_dict(state_dict, strict=False)

        missing = load_result.missing_keys if hasattr(load_result, 'missing_keys') else []
        unexpected = load_result.unexpected_keys if hasattr(load_result, 'unexpected_keys') else []

        print(f"  → Loaded: {label}")
        print(f"     Missing keys:    {len(missing)}")
        print(f"     Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print(f"     (first few missing): {missing[:3]}")
        if len(unexpected) > 0:
            print(f"     (first few unexpected): {unexpected[:3]}")

        return True

    # Priority 1: DINO checkpoint with teacher
    if os.path.exists("dino_best_checkpoint.pth") and not encoder_loaded:
        try:
            print("\n✓ Attempting to load DINO checkpoint...")
            # checkpoint = torch.load("dino_best_checkpoint.pth", map_location=device)
            checkpoint = torch.load("dino_best_checkpoint.pth", map_location=device, weights_only=False)

            if "teacher_state_dict" in checkpoint:
                teacher_state = checkpoint["teacher_state_dict"]
            elif "state_dict" in checkpoint:
                teacher_state = checkpoint["state_dict"]
            else:
                raise KeyError("No 'teacher_state_dict' or 'state_dict' found")

            epoch_info = checkpoint.get('epoch', '?')
            loss_info = checkpoint.get('loss', 0)

            encoder_loaded = load_encoder_state(
                encoder,
                teacher_state,
                prefix_to_strip="backbone.",
                label=f"DINO Teacher (epoch {epoch_info}, loss {loss_info:.4f})"
            )
            encoder_source = f"DINO Teacher checkpoint (epoch {epoch_info})"

        except Exception as e:
            print(f"⚠ Failed to load DINO checkpoint: {e}")

    # Priority 2: Direct teacher encoder
    if not encoder_loaded and os.path.exists("rad_dino_teacher_encoder.pth"):
        try:
            print("\n✓ Attempting to load teacher encoder (direct file)...")
            # teacher_state = torch.load("rad_dino_teacher_encoder.pth", map_location=device)
            teacher_state = torch.load("rad_dino_teacher_encoder.pth", map_location=device, weights_only=False)

            # Handle nested state dicts
            if isinstance(teacher_state, dict):
                for key in ["encoder_state_dict", "state_dict", "model_state_dict"]:
                    if key in teacher_state:
                        teacher_state = teacher_state[key]
                        print(f"  Found nested state dict under key '{key}'")
                        break

            encoder_loaded = load_encoder_state(
                encoder,
                teacher_state,
                prefix_to_strip=None,
                label="DINO Teacher (direct file)"
            )
            encoder_source = "DINO Teacher (direct file)"

        except Exception as e:
            print(f"⚠ Failed to load teacher encoder: {e}")

    # Priority 3: Direct student encoder
    if not encoder_loaded and os.path.exists("rad_dino_student_encoder.pth"):
        try:
            print("\n✓ Attempting to load student encoder (direct file)...")
            # student_state = torch.load("rad_dino_student_encoder.pth", map_location=device)
            student_state = torch.load("rad_dino_student_encoder.pth", map_location=device, weights_only=False)
            # Handle nested state dicts
            if isinstance(student_state, dict):
                for key in ["encoder_state_dict", "state_dict", "model_state_dict"]:
                    if key in student_state:
                        student_state = student_state[key]
                        print(f"  Found nested state dict under key '{key}'")
                        break

            encoder_loaded = load_encoder_state(
                encoder,
                student_state,
                prefix_to_strip=None,
                label="DINO Student (direct file)"
            )
            encoder_source = "DINO Student (direct file)"
            print("  ⚠ Note: Teacher encoder usually performs better")

        except Exception as e:
            print(f"⚠ Failed to load student encoder: {e}")

    # Fallback
    if not encoder_loaded:
        print("\n⚠ WARNING: Using base RAD-DINO (not fine-tuned on your data)")
        print("  Looking for any of:")
        print("    1. dino_best_checkpoint.pth")
        print("    2. rad_dino_teacher_encoder.pth")
        print("    3. rad_dino_student_encoder.pth")

    print(f"\n✓ Encoder source: {encoder_source}\n")

    # Set to eval mode
    encoder.eval()

    # Create prototypical network
    model = PrototypicalNetwork(
        encoder=encoder,
        embedding_dim=768,
        projection_dim=256,
        freeze_encoder=True
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %:          {100 * trainable_params / total_params:.2f}%")

    # ========================================================================
    # Training Configuration
    # ========================================================================

    num_episodes = 5000
    val_episodes = 300
    val_interval = 200
    learning_rate = 5e-5

    print(f"\nTraining Configuration:")
    print(f"  {n_way}-way {n_support}-shot")
    print(f"  Query per class: {n_query}")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Validation episodes: {val_episodes}")
    print(f"  Validation interval: {val_interval}")
    print(f"  Learning rate: {learning_rate}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=1e-6
    )

    # ========================================================================
    # Training Loop
    # ========================================================================

    print(f"\n{'=' * 70}")
    print("Starting 4-Way Few-Shot Training")
    print(f"{'=' * 70}\n")

    best_val_acc = 0.0
    running_loss = 0.0
    running_acc = 0.0

    try:
        for episode in range(1, num_episodes + 1):
            loss, acc = train_episode(
                model, train_dataset, n_way, n_support, n_query, optimizer, device
            )
            scheduler.step()

            running_loss += loss
            running_acc += acc

            if episode % 50 == 0:
                avg_loss = running_loss / 50
                avg_acc = running_acc / 50
                print(f"Episode {episode}/{num_episodes} | "
                      f"Loss: {avg_loss:.4f} | Acc: {avg_acc * 100:.2f}% | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                running_loss = 0.0
                running_acc = 0.0

            if episode % val_interval == 0:
                print(f"\n{'=' * 70}")
                print(f"Validation at Episode {episode}")
                print(f"{'=' * 70}")

                val_loss, val_acc = validate_episode(
                    model, val_dataset, n_way, n_support, n_query, device, val_episodes
                )

                print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc * 100:.2f}%")
                print(f"Random baseline: {100.0 / n_way:.2f}%\n")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'episode': episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': best_val_acc,
                        'n_way': n_way,
                        'n_support': n_support,
                        'class_names': class_names,
                    }, 'chexpert_4way_best.pth')
                    print(f"✓ Saved best model (Val Acc: {best_val_acc * 100:.2f}%)\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Final Results
    # ========================================================================

    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Random Baseline: {100.0 / n_way:.2f}%")
    print(f"Improvement over random: {(best_val_acc * 100 - 100.0 / n_way):.2f}%")
    print(f"\nClass Names:")
    for idx, name in class_names.items():
        print(f"  {idx}: {name}")

    torch.save(model.state_dict(), 'chexpert_4way_final.pth')
    print(f"\n✓ Saved final model\n")


if __name__ == '__main__':
    main()