import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import copy


# ============================================================================
# DINO HEAD (for self-distillation)
# ============================================================================

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ============================================================================
# MULTI-CROP AUGMENTATION (FIXED WITH RAD-DINO NORMALIZATION)
# ============================================================================

class MultiCropAugmentation:
    def __init__(self, image_processor, global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4), n_local_crops=8,
                 size=224, local_crop_size=96):
        """
        DINOv2-style multi-crop augmentation with RAD-DINO normalization

        Args:
            image_processor: AutoImageProcessor from RAD-DINO
            global_crops_scale: Scale range for global crops
            local_crops_scale: Scale range for local crops
            n_local_crops: Number of local crops
            size: Size of global crops
            local_crop_size: Size of local crops
        """
        self.n_local_crops = n_local_crops

        # ✅ FIX: Use RAD-DINO's own normalization stats
        rad_mean = image_processor.image_mean
        rad_std = image_processor.image_std

        print(f"✓ Using RAD-DINO normalization:")
        print(f"  Mean: {rad_mean}")
        print(f"  Std: {rad_std}")

        normalize = transforms.Normalize(mean=rad_mean, std=rad_std)

        # Global crop 1 (strong augmentation)
        self.global_crop1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale,
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=1.0),
            transforms.ToTensor(),
            normalize
        ])

        # Global crop 2 (strong augmentation with solarization)
        self.global_crop2 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale,
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            normalize
        ])

        # Local crops (smaller patches)
        self.local_crop = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale,
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, image):
        """
        ✅ Handles different input sizes automatically
        Your images: 320x369 and 390x320 are both handled by RandomResizedCrop
        """
        crops = []

        # 2 global crops
        crops.append(self.global_crop1(image))
        crops.append(self.global_crop2(image))

        # N local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local_crop(image))

        return crops


# ============================================================================
# DINO DATASET
# ============================================================================

class DINODataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        crops = self.transform(image)  # Returns list of crops
        return crops


# ============================================================================
# DINO STUDENT-TEACHER MODEL (FIXED WITH pooler_output)
# ============================================================================

class DINOModel(nn.Module):
    def __init__(self, backbone, unfreeze_last_n=2, embed_dim=768,
                 out_dim=65536, use_bn_in_head=False, norm_last_layer=True):
        super().__init__()

        self.backbone = backbone
        self.embed_dim = embed_dim

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ✅ FIX: Check if encoder.layer exists (handle different architectures)
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            num_layers = len(self.backbone.encoder.layer)
            print(f"✓ Found encoder with {num_layers} layers")

            # Unfreeze last N layers
            for i in range(num_layers - unfreeze_last_n, num_layers):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = True
                print(f"  ✓ Unfrozen layer {i}")
        else:
            print("⚠ Warning: Could not find encoder.layer structure")
            print("  Attempting alternative attribute names...")

            # Try alternative structures
            if hasattr(self.backbone, 'layers'):
                num_layers = len(self.backbone.layers)
                for i in range(num_layers - unfreeze_last_n, num_layers):
                    for param in self.backbone.layers[i].parameters():
                        param.requires_grad = True
                    print(f"  ✓ Unfrozen layer {i}")

        # DINO projection head
        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer
        )

    # def forward(self, x):
    #     """
    #     ✅ FIX: Use pooler_output instead of last_hidden_state[:, 0, :]
    #     This is the official RAD-DINO recommended way
    #     """
    #     outputs = self.backbone(x)
    #
    #     # Use pooler_output (official RAD-DINO way)
    #     if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
    #         features = outputs.pooler_output  # [batch, 768]
    #     else:
    #         # Fallback to CLS token if pooler_output not available
    #         features = outputs.last_hidden_state[:, 0, :]
    #
    #     # Project through DINO head
    #     return self.head(features)
    def forward(self, x):
        """Handle variable input sizes"""
        # Resize to 224x224 if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        outputs = self.backbone(x)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]

        return self.head(features)


# ============================================================================
# DINO LOSS
# ============================================================================

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Temperature schedule for teacher
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of student and teacher
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output"""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ============================================================================
# EMA TEACHER UPDATE
# ============================================================================

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    """Exponential Moving Average update for teacher model"""
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """Cosine schedule for momentum/learning rate"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# # ============================================================================
# # TRAINING FUNCTION (FIXED WITH PROPER FP16 AND PER-ITERATION LR)
# # ============================================================================
#
# def train_dino_epoch(student, teacher, loader, criterion, optimizer,
#                      lr_schedule, momentum_schedule, epoch, device, fp16_scaler=None):
#     """
#     ✅ FIXED:
#     - Added per-iteration LR update
#     - Added proper autocast for mixed precision
#     """
#     student.train()
#     teacher.eval()
#
#     total_loss = 0
#
#     for it, crops in enumerate(tqdm(loader, desc=f'Epoch {epoch}')):
#         # ✅ FIX: Per-iteration learning rate update
#         it_global = epoch * len(loader) + it
#
#         # Update learning rate per iteration (not per epoch)
#         lr = lr_schedule[it_global]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#
#         # Update teacher momentum per iteration
#         momentum = momentum_schedule[it_global]
#
#         # Move crops to device
#         crops = [crop.to(device) for crop in crops]
#
#         # ✅ FIX: Proper mixed precision with autocast
#         if fp16_scaler is None:
#             # FP32 path
#             teacher_output = teacher(torch.cat(crops[:2]))
#             student_output = student(torch.cat(crops))
#             loss = criterion(student_output, teacher_output, epoch)
#
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
#             optimizer.step()
#         else:
#             # Mixed precision path with autocast
#             with torch.cuda.amp.autocast():
#                 teacher_output = teacher(torch.cat(crops[:2]))
#                 student_output = student(torch.cat(crops))
#                 loss = criterion(student_output, teacher_output, epoch)
#
#             # Backward pass with scaler
#             optimizer.zero_grad()
#             fp16_scaler.scale(loss).backward()
#             fp16_scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
#             fp16_scaler.step(optimizer)
#             fp16_scaler.update()
#
#         # EMA update for teacher
#         update_teacher(student, teacher, momentum)
#
#         total_loss += loss.item()
#
#     return total_loss / len(loader)


# ============================================================================
# TRAINING FUNCTION (FIXED - Process crops individually)
# ============================================================================

def train_dino_epoch(student, teacher, loader, criterion, optimizer,
                     lr_schedule, momentum_schedule, epoch, device, fp16_scaler=None):
    """
    ✅ FIXED: Process crops individually to handle different sizes
    """
    student.train()
    teacher.eval()

    total_loss = 0

    for it, crops in enumerate(tqdm(loader, desc=f'Epoch {epoch}')):
        # ✅ FIX: Per-iteration learning rate update
        it_global = epoch * len(loader) + it

        # Update learning rate per iteration (not per epoch)
        lr = lr_schedule[it_global]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Update teacher momentum per iteration
        momentum = momentum_schedule[it_global]

        # Move crops to device
        crops = [crop.to(device) for crop in crops]

        # ✅ FIX: Proper mixed precision with autocast
        if fp16_scaler is None:
            # FP32 path
            # Process teacher (only global crops - first 2)
            teacher_outputs = []
            for crop in crops[:2]:
                teacher_outputs.append(teacher(crop))
            teacher_output = torch.cat(teacher_outputs, dim=0)

            # Process student (all crops)
            student_outputs = []
            for crop in crops:
                student_outputs.append(student(crop))
            student_output = torch.cat(student_outputs, dim=0)

            loss = criterion(student_output, teacher_output, epoch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            optimizer.step()
        else:
            # Mixed precision path with autocast
            with torch.cuda.amp.autocast():
                # Process teacher (only global crops)
                teacher_outputs = []
                for crop in crops[:2]:
                    teacher_outputs.append(teacher(crop))
                teacher_output = torch.cat(teacher_outputs, dim=0)

                # Process student (all crops)
                student_outputs = []
                for crop in crops:
                    student_outputs.append(student(crop))
                student_output = torch.cat(student_outputs, dim=0)

                loss = criterion(student_output, teacher_output, epoch)

            # Backward pass with scaler
            optimizer.zero_grad()
            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for teacher
        update_teacher(student, teacher, momentum)

        total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 4  # Small batch size
    num_epochs = 50
    unfreeze_last_n = 2

    # DINO parameters
    out_dim = 65536
    warmup_teacher_temp = 0.04
    teacher_temp = 0.04
    warmup_teacher_temp_epochs = 0
    student_temp = 0.1
    center_momentum = 0.9

    # Momentum teacher
    momentum_teacher = 0.996
    final_momentum_teacher = 1.0

    # Learning rate
    base_lr = 0.0005
    final_lr = 1e-6
    warmup_epochs = 10

    # Multi-crop
    n_local_crops = 8
    global_crop_size = 224
    local_crop_size = 96

    # Mixed precision
    use_fp16 = True

    print("=" * 70)
    print("DINOv2-style Self-Distillation for RAD-DINO (OPTIMIZED)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Unfreezing last {unfreeze_last_n} layers")
    print(f"Global crops: 2 x {global_crop_size}x{global_crop_size}")
    print(f"Local crops: {n_local_crops} x {local_crop_size}x{local_crop_size}")
    print(f"Mixed precision (FP16): {use_fp16}")
    print("=" * 70)

    # ========================================================================
    # Load RAD-DINO and Image Processor
    # ========================================================================

    print("\n✓ Loading RAD-DINO model and image processor...")
    backbone = AutoModel.from_pretrained("microsoft/rad-dino")
    image_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    embed_dim = backbone.config.hidden_size

    print(f"  Embedding dimension: {embed_dim}")

    # ========================================================================
    # Data with RAD-DINO Normalization
    # ========================================================================

    # ✅ Multi-crop augmentation with RAD-DINO's normalization
    transform = MultiCropAugmentation(
        image_processor=image_processor,  # Pass the processor
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        n_local_crops=n_local_crops,
        size=global_crop_size,
        local_crop_size=local_crop_size
    )

    # ✅ Your images with different sizes are handled automatically
    print("\n✓ Your image sizes (320x369, 390x320) are handled automatically")
    print("  RandomResizedCrop will resize all to fixed sizes (224x224, 96x96)")

    image_paths=[]
    with open("file_names.txt","r") as f:
        txt=f.read()
        image_paths=txt.split(",")


    dataset = DINODataset(image_paths[:15000], transform)

    # Custom collate function for multi-crop
    def collate_fn(batch):
        return [torch.stack([img[i] for img in batch]) for i in range(2 + n_local_crops)]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    print(f"\nDataset: {len(dataset)} images")
    print(f"Batches per epoch: {len(loader)}\n")

    # ========================================================================
    # Models
    # ========================================================================

    # Student model
    student = DINOModel(
        backbone=copy.deepcopy(backbone),
        unfreeze_last_n=unfreeze_last_n,
        embed_dim=embed_dim,
        out_dim=out_dim,
        use_bn_in_head=False,
        norm_last_layer=True
    ).to(device)

    # Teacher model (EMA of student)
    teacher = DINOModel(
        backbone=copy.deepcopy(backbone),
        unfreeze_last_n=unfreeze_last_n,
        embed_dim=embed_dim,
        out_dim=out_dim,
        use_bn_in_head=False,
        norm_last_layer=True
    ).to(device)

    # Teacher doesn't require gradients
    for param in teacher.parameters():
        param.requires_grad = False

    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # ========================================================================
    # Optimizer and Schedulers
    # ========================================================================

    params_to_optimize = [p for p in student.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=base_lr,
        weight_decay=0.04
    )

    # ✅ FIX: Per-iteration learning rate schedule
    lr_schedule = cosine_scheduler(
        base_value=base_lr,
        final_value=final_lr,
        epochs=num_epochs,
        niter_per_ep=len(loader),
        warmup_epochs=warmup_epochs
    )

    # Momentum schedule for teacher
    momentum_schedule = cosine_scheduler(
        base_value=momentum_teacher,
        final_value=final_momentum_teacher,
        epochs=num_epochs,
        niter_per_ep=len(loader)
    )

    print(f"\nLearning rate schedule: {base_lr:.6f} → {final_lr:.6f}")
    print(f"Teacher momentum schedule: {momentum_teacher:.4f} → {final_momentum_teacher:.4f}")

    # Loss
    criterion = DINOLoss(
        out_dim=out_dim,
        ncrops=2 + n_local_crops,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        nepochs=num_epochs,
        student_temp=student_temp,
        center_momentum=center_momentum
    ).to(device)

    # ✅ FIX: Proper mixed precision scaler
    fp16_scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # ========================================================================
    # Training Loop
    # ========================================================================

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # Train one epoch with proper LR scheduling
        loss = train_dino_epoch(
            student=student,
            teacher=teacher,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_schedule=lr_schedule,  # Pass the schedule
            momentum_schedule=momentum_schedule,
            epoch=epoch,
            device=device,
            fp16_scaler=fp16_scaler
        )

        # Get current LR (last iteration of epoch)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {loss:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'dino_best_checkpoint.pth')
            print(f"  ✓ Saved best checkpoint (Loss: {best_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'dino_checkpoint_epoch{epoch + 1}.pth')
            print(f"  ✓ Saved checkpoint at epoch {epoch + 1}")

    # ========================================================================
    # Save Final Models
    # ========================================================================

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Save student encoder
    torch.save(student.backbone.state_dict(), 'rad_dino_student_encoder.pth')
    print("✓ Saved fine-tuned student encoder")

    # Save teacher encoder (recommended for downstream tasks)
    torch.save(teacher.backbone.state_dict(), 'rad_dino_teacher_encoder.pth')
    print("✓ Saved fine-tuned teacher encoder (RECOMMENDED)")

    print(f"\nBest Loss: {best_loss:.4f}")
    print("\nNext steps:")
    print("1. Use 'rad_dino_teacher_encoder.pth' for few-shot learning")
    print("2. Load this encoder into your Prototypical Network")


if __name__ == '__main__':
    main()