"""
Evaluation metrics for audio classification.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'f1_score': f1_score(all_targets, all_predictions, average='weighted'),
        'precision': precision_score(all_targets, all_predictions, average='weighted'),
        'recall': recall_score(all_targets, all_predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(all_targets, all_predictions)
    }
    
    return metrics


def print_evaluation_report(metrics):
    """
    Print formatted evaluation report.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("="*50 + "\n")
