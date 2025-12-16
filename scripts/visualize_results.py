"""
Visualize training results and model performance.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config


def plot_confusion_matrix_detailed(cm, class_names, output_path):
    """Plot detailed confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix")


def plot_per_class_accuracy(cm, class_names, output_path):
    """Plot per-class accuracy."""
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_acc, color='steelblue', alpha=0.8)
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if per_class_acc[i] >= 0.8:
            bar.set_color('green')
        elif per_class_acc[i] >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Animal Category', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Classification Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (≥80%)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (≥60%)')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class accuracy")


def plot_training_curves(history_file, output_path):
    """Plot training and validation curves."""
    if not os.path.exists(history_file):
        print(f"⚠ Training history not found at {history_file}")
        return
    
    history = np.load(history_file, allow_pickle=True).item()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves")


def plot_misclassification_analysis(cm, class_names, output_path):
    """Analyze and visualize common misclassifications."""
    # Find top misclassifications
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j],
                    'rate': cm[i, j] / cm[i].sum()
                })
    
    # Sort by count
    misclassifications = sorted(misclassifications, key=lambda x: x['count'], reverse=True)[:10]
    
    if not misclassifications:
        print("⚠ No misclassifications found")
        return
    
    # Plot
    labels = [f"{m['true']} → {m['pred']}" for m in misclassifications]
    counts = [m['count'] for m in misclassifications]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(labels, counts, color='coral', alpha=0.8)
    plt.xlabel('Number of Misclassifications', fontsize=12)
    plt.ylabel('True → Predicted', fontsize=12)
    plt.title('Top 10 Misclassification Pairs', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved misclassification analysis")


def generate_classification_report_viz(cm, class_names, output_path):
    """Generate visual classification report."""
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        np.repeat(range(len(class_names)), cm.sum(axis=1).astype(int)),
        np.repeat(range(len(class_names)), cm.sum(axis=1).astype(int)),
        labels=range(len(class_names)),
        zero_division=0
    )
    
    # Recalculate from confusion matrix
    precision = np.diag(cm) / cm.sum(axis=0)
    recall = np.diag(cm) / cm.sum(axis=1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='orange')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='green')
    
    ax.set_xlabel('Animal Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved classification metrics")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/results',
                        help='Output directory for result visualizations')
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING RESULTS VISUALIZATIONS")
    print("="*60 + "\n")
    
    config = load_config(args.config)
    animal_categories = sorted(config['data']['animal_categories'])
    
    # Note: These visualizations require actual training results
    print("⚠ Note: These visualizations require training to be completed first.")
    print("   After training, confusion matrix and metrics will be generated.")
    print(f"\n✓ Output directory created: {output_dir}")
    print("\nRun this script again after training completes.")


if __name__ == '__main__':
    main()
