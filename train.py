"""
Training loop for Few-Shot Learning with Prototypical Networks
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path


class Trainer:
    """Trainer class for few-shot learning"""
    
    def __init__(self, model, config, train_dataset, val_dataset):
        """
        Args:
            model: Prototypical Network model
            config: Configuration object
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.DEVICE
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.SCHEDULER_STEP,
            gamma=config.SCHEDULER_GAMMA
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Initialize wandb if enabled
        if config.USE_WANDB:
            wandb.init(
                project=config.WANDB_PROJECT,
                config={
                    "backbone": config.BACKBONE,
                    "n_way": config.N_WAY,
                    "k_shot": config.K_SHOT,
                    "learning_rate": config.LEARNING_RATE,
                    "embedding_dim": config.EMBEDDING_DIM,
                }
            )
            wandb.watch(self.model)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_accs = []
        
        pbar = tqdm(range(self.config.NUM_EPISODES), desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for episode_idx in pbar:
            # Sample episode
            support_images, support_labels, query_images, query_labels = \
                self.train_dataset.sample_episode(
                    self.config.N_WAY,
                    self.config.K_SHOT,
                    self.config.N_QUERY
                )
            
            # Move to device
            support_images = support_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_images = query_images.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                support_images, support_labels, query_images,
                self.config.N_WAY, self.config.K_SHOT
            )
            
            # Compute loss
            loss = F.cross_entropy(logits, query_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
            
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{np.mean(epoch_losses):.4f}',
                'acc': f'{np.mean(epoch_accs):.4f}'
            })
            
            # Log to wandb
            if self.config.USE_WANDB and episode_idx % self.config.LOG_INTERVAL == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy,
                    "train/epoch": epoch,
                })
        
        # Step scheduler
        self.scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        
        return avg_loss, avg_acc
    
    def evaluate(self, dataset, n_episodes=None):
        """
        Evaluate on dataset
        Args:
            dataset: Dataset to evaluate on
            n_episodes: Number of episodes (if None, use config.NUM_EVAL_EPISODES)
        Returns:
            avg_accuracy: Average accuracy
            confidence_interval: 95% confidence interval
        """
        if n_episodes is None:
            n_episodes = self.config.NUM_EVAL_EPISODES
        
        self.model.eval()
        accuracies = []
        
        with torch.no_grad():
            for _ in tqdm(range(n_episodes), desc="Evaluating"):
                # Sample episode
                support_images, support_labels, query_images, query_labels = \
                    dataset.sample_episode(
                        self.config.N_WAY,
                        self.config.K_SHOT,
                        self.config.N_QUERY
                    )
                
                # Move to device
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                # Forward pass
                logits = self.model(
                    support_images, support_labels, query_images,
                    self.config.N_WAY, self.config.K_SHOT
                )
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                accuracies.append(accuracy)
        
        # Compute statistics
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        confidence_interval = 1.96 * std_accuracy / np.sqrt(n_episodes)
        
        return avg_accuracy, confidence_interval
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training on {self.device}")
        print(f"Backbone: {self.config.BACKBONE}")
        print(f"Embedding dimension: {self.config.EMBEDDING_DIM}")
        print(f"Training: {self.config.N_WAY}-way {self.config.K_SHOT}-shot")
        print(f"Episodes per epoch: {self.config.NUM_EPISODES}")
        print()
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_acc, val_ci = self.evaluate(self.val_dataset)
            self.val_accs.append(val_acc)
            
            # Print results
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc:  {train_acc:.4f}")
            print(f"  Val Acc:    {val_acc:.4f} ± {val_ci:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "train/epoch_accuracy": train_acc,
                    "val/accuracy": val_acc,
                    "val/confidence_interval": val_ci,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best model! Val Acc: {val_acc:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.best_val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        
        if is_best:
            path = self.config.CHECKPOINT_DIR / "best_model.pth"
        else:
            path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']


class AdvancedEvaluator:
    """Advanced evaluation across multiple scenarios"""
    
    def __init__(self, model, config, dataset):
        """
        Args:
            model: Trained model
            config: Configuration object
            dataset: Dataset to evaluate on
        """
        self.model = model.to(config.DEVICE)
        self.config = config
        self.dataset = dataset
        self.device = config.DEVICE
        self.model.eval()
    
    def evaluate_multiple_shots(self, n_way=5, k_shots=[1, 5, 10, 20], n_query=15, n_episodes=600):
        """
        Evaluate performance across different k-shot scenarios
        Args:
            n_way: Number of classes
            k_shots: List of k-shot values to test
            n_query: Number of query examples
            n_episodes: Number of episodes per scenario
        Returns:
            results: Dictionary of results
        """
        results = {}
        
        print(f"\nEvaluating {n_way}-way classification with different shots:")
        
        for k_shot in k_shots:
            print(f"\n{k_shot}-shot learning:")
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
                    
                    # Forward pass
                    logits = self.model(
                        support_images, support_labels, query_images,
                        n_way, k_shot
                    )
                    
                    # Compute accuracy
                    predictions = torch.argmax(logits, dim=1)
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
    
    def evaluate_multiple_ways(self, n_ways=[2, 5, 10], k_shot=5, n_query=15, n_episodes=600):
        """
        Evaluate performance across different n-way scenarios
        Args:
            n_ways: List of n-way values to test
            k_shot: Number of support examples
            n_query: Number of query examples
            n_episodes: Number of episodes per scenario
        Returns:
            results: Dictionary of results
        """
        results = {}
        
        print(f"\nEvaluating {k_shot}-shot learning with different ways:")
        
        for n_way in n_ways:
            # Skip if dataset doesn't have enough classes
            if n_way > len(self.dataset.classes):
                print(f"  Skipping {n_way}-way (only {len(self.dataset.classes)} classes available)")
                continue
            
            print(f"\n{n_way}-way classification:")
            accuracies = []
            
            with torch.no_grad():
                for _ in tqdm(range(n_episodes), desc=f"  {n_way}-way"):
                    # Sample episode
                    support_images, support_labels, query_images, query_labels = \
                        self.dataset.sample_episode(n_way, k_shot, n_query)
                    
                    # Move to device
                    support_images = support_images.to(self.device)
                    support_labels = support_labels.to(self.device)
                    query_images = query_images.to(self.device)
                    query_labels = query_labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(
                        support_images, support_labels, query_images,
                        n_way, k_shot
                    )
                    
                    # Compute accuracy
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_labels).float().mean().item()
                    accuracies.append(accuracy)
            
            # Compute statistics
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            ci = 1.96 * std_acc / np.sqrt(n_episodes)
            
            results[f"{n_way}-way"] = {
                'accuracy': avg_acc,
                'std': std_acc,
                'ci': ci,
                'all_accuracies': accuracies
            }
            
            print(f"  Accuracy: {avg_acc:.4f} ± {ci:.4f}")
        
        return results


if __name__ == "__main__":
    print("Training module loaded successfully")

