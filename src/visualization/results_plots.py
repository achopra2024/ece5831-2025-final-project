"""
Result visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names
        title (str): Plot title
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy']):
    """
    Plot training history.
    
    Args:
        history (dict): Training history dictionary
        metrics (list): Metrics to plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        axes[idx].plot(history[f'train_{metric}'], label=f'Train {metric}')
        axes[idx].plot(history[f'val_{metric}'], label=f'Val {metric}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_title(f'{metric.capitalize()} over Epochs')
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, class_names=None, title='Class Distribution'):
    """
    Plot distribution of classes.
    
    Args:
        labels (np.ndarray): Array of labels
        class_names (list): List of class names
        title (str): Plot title
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(14, 6))
    plt.bar(unique, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    if class_names:
        plt.xticks(unique, [class_names[i] for i in unique], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
