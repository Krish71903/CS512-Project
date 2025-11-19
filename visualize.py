"""
Comprehensive visualization and metrics for Few-Shot Learning results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class Visualizer:
    """Comprehensive visualization suite for few-shot learning"""
    
    def __init__(self, results_dir):
        """
        Args:
            results_dir: Directory to save visualizations
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_training_curves(self, train_losses, train_accs, val_accs, save_name="training_curves.png"):
        """
        Plot training and validation curves
        Args:
            train_losses: List of training losses per epoch
            train_accs: List of training accuracies per epoch
            val_accs: List of validation accuracies per epoch
            save_name: Filename to save plot
        """
        epochs = range(1, len(train_losses) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
        plt.close()
    
    def plot_shot_comparison(self, results, n_way=5, save_name="shot_comparison.png"):
        """
        Plot accuracy comparison across different k-shot scenarios
        Args:
            results: Dictionary with results for different shots
            n_way: Number of ways
            save_name: Filename to save plot
        """
        shots = []
        accuracies = []
        errors = []
        
        for key in sorted(results.keys(), key=lambda x: int(x.split('-')[0])):
            shot = int(key.split('-')[0])
            shots.append(shot)
            accuracies.append(results[key]['accuracy'] * 100)
            errors.append(results[key]['ci'] * 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar plot with error bars
        bars = ax.bar(range(len(shots)), accuracies, yerr=errors, capsize=10,
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize plot
        ax.set_xlabel('Number of Support Examples (K-shot)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{n_way}-Way Classification: Performance vs K-Shot', 
                     fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(shots)))
        ax.set_xticklabels([f'{s}-shot' for s in shots])
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, acc, err) in enumerate(zip(bars, accuracies, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 1,
                   f'{acc:.2f}%\n±{err:.2f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved shot comparison to {save_path}")
        plt.close()
    
    def plot_way_comparison(self, results, k_shot=5, save_name="way_comparison.png"):
        """
        Plot accuracy comparison across different n-way scenarios
        Args:
            results: Dictionary with results for different ways
            k_shot: Number of shots
            save_name: Filename to save plot
        """
        ways = []
        accuracies = []
        errors = []
        
        for key in sorted(results.keys(), key=lambda x: int(x.split('-')[0])):
            way = int(key.split('-')[0])
            ways.append(way)
            accuracies.append(results[key]['accuracy'] * 100)
            errors.append(results[key]['ci'] * 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Line plot with error bands
        ax.plot(ways, accuracies, 'o-', color='steelblue', linewidth=3, 
                markersize=10, label='Accuracy')
        ax.fill_between(ways, 
                        np.array(accuracies) - np.array(errors),
                        np.array(accuracies) + np.array(errors),
                        alpha=0.3, color='steelblue')
        
        # Customize plot
        ax.set_xlabel('Number of Classes (N-way)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{k_shot}-Shot Learning: Performance vs N-Way', 
                     fontsize=16, fontweight='bold')
        ax.set_xticks(ways)
        ax.set_xticklabels([f'{w}-way' for w in ways])
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add value labels
        for way, acc, err in zip(ways, accuracies, errors):
            ax.text(way, acc + err + 2, f'{acc:.2f}%\n±{err:.2f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved way comparison to {save_path}")
        plt.close()
    
    def plot_accuracy_distribution(self, results, save_name="accuracy_distribution.png"):
        """
        Plot distribution of accuracies for different scenarios
        Args:
            results: Dictionary with results containing 'all_accuracies'
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (key, data) in enumerate(results.items()):
            if idx >= len(axes):
                break
            
            accuracies = np.array(data['all_accuracies']) * 100
            
            # Histogram
            axes[idx].hist(accuracies, bins=30, color='steelblue', 
                          alpha=0.7, edgecolor='black')
            
            # Add mean line
            mean_acc = data['accuracy'] * 100
            axes[idx].axvline(mean_acc, color='red', linestyle='--', 
                             linewidth=2, label=f'Mean: {mean_acc:.2f}%')
            
            # Customize
            axes[idx].set_xlabel('Accuracy (%)', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'Accuracy Distribution: {key}', 
                               fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy distribution to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            save_name="confusion_matrix.png"):
        """
        Plot confusion matrix
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_name: Filename to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_embedding_visualization(self, embeddings, labels, class_names=None,
                                    save_name="embedding_visualization.png"):
        """
        Visualize embeddings using t-SNE
        Args:
            embeddings: Embedding vectors [N, D]
            labels: Labels [N]
            class_names: List of class names
            save_name: Filename to save plot
        """
        print("Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each class with different color
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            class_name = class_names[label] if class_names else f'Class {label}'
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color], label=class_name, alpha=0.6, s=50,
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Embedding Space Visualization (t-SNE)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved embedding visualization to {save_path}")
        plt.close()
    
    def create_results_summary(self, results, save_name="results_summary.csv"):
        """
        Create a summary CSV of all results
        Args:
            results: Dictionary with results
            save_name: Filename to save CSV
        """
        summary_data = []
        
        for key, data in results.items():
            summary_data.append({
                'Scenario': key,
                'Accuracy (%)': f"{data['accuracy'] * 100:.2f}",
                'Std Dev (%)': f"{data['std'] * 100:.2f}",
                '95% CI (%)': f"±{data['ci'] * 100:.2f}",
            })
        
        df = pd.DataFrame(summary_data)
        save_path = self.results_dir / save_name
        df.to_csv(save_path, index=False)
        print(f"\nResults Summary:")
        print(df.to_string(index=False))
        print(f"\nSaved results summary to {save_path}")
    
    def plot_comprehensive_results(self, shot_results, way_results=None):
        """
        Create a comprehensive results figure with multiple subplots
        Args:
            shot_results: Results from different k-shot evaluations
            way_results: Results from different n-way evaluations
        """
        if way_results:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. K-shot comparison (bar plot)
        ax1 = fig.add_subplot(gs[0, 0])
        shots = []
        shot_accs = []
        shot_errs = []
        for key in sorted(shot_results.keys(), key=lambda x: int(x.split('-')[0])):
            shot = int(key.split('-')[0])
            shots.append(shot)
            shot_accs.append(shot_results[key]['accuracy'] * 100)
            shot_errs.append(shot_results[key]['ci'] * 100)
        
        bars = ax1.bar(range(len(shots)), shot_accs, yerr=shot_errs, 
                       capsize=5, color='steelblue', alpha=0.8)
        ax1.set_xlabel('K-Shot', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Performance vs K-Shot', fontweight='bold', fontsize=14)
        ax1.set_xticks(range(len(shots)))
        ax1.set_xticklabels([f'{s}' for s in shots])
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Accuracy distribution for best shot
        ax2 = fig.add_subplot(gs[0, 1])
        best_shot_key = max(shot_results.keys(), 
                           key=lambda x: shot_results[x]['accuracy'])
        best_shot_accs = np.array(shot_results[best_shot_key]['all_accuracies']) * 100
        ax2.hist(best_shot_accs, bins=30, color='steelblue', 
                alpha=0.7, edgecolor='black')
        mean_acc = shot_results[best_shot_key]['accuracy'] * 100
        ax2.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_acc:.2f}%')
        ax2.set_xlabel('Accuracy (%)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title(f'Accuracy Distribution ({best_shot_key})', 
                     fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Box plot for all shots
        ax3 = fig.add_subplot(gs[1, :])
        shot_data = []
        shot_labels = []
        for key in sorted(shot_results.keys(), key=lambda x: int(x.split('-')[0])):
            shot_data.append(np.array(shot_results[key]['all_accuracies']) * 100)
            shot_labels.append(key)
        
        bp = ax3.boxplot(shot_data, labels=shot_labels, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        ax3.set_xlabel('Scenario', fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontweight='bold')
        ax3.set_title('Accuracy Distribution Across Scenarios', 
                     fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 100])
        
        # 4. N-way comparison (if available)
        if way_results:
            ax4 = fig.add_subplot(gs[2, :])
            ways = []
            way_accs = []
            way_errs = []
            for key in sorted(way_results.keys(), key=lambda x: int(x.split('-')[0])):
                way = int(key.split('-')[0])
                ways.append(way)
                way_accs.append(way_results[key]['accuracy'] * 100)
                way_errs.append(way_results[key]['ci'] * 100)
            
            ax4.plot(ways, way_accs, 'o-', color='steelblue', 
                    linewidth=3, markersize=10)
            ax4.fill_between(ways, 
                            np.array(way_accs) - np.array(way_errs),
                            np.array(way_accs) + np.array(way_errs),
                            alpha=0.3, color='steelblue')
            ax4.set_xlabel('N-Way', fontweight='bold')
            ax4.set_ylabel('Accuracy (%)', fontweight='bold')
            ax4.set_title('Performance vs N-Way', fontweight='bold', fontsize=14)
            ax4.set_xticks(ways)
            ax4.set_ylim([0, 100])
            ax4.grid(True, alpha=0.3)
        
        save_path = self.results_dir / "comprehensive_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive results to {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Visualization module loaded successfully")

