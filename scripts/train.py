"""
Training script for audio classification models.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn import AudioCNN
from src.models.lstm import AudioLSTM
from src.models.improved_cnn import ImprovedAudioCNN, AttentionAudioCNN
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def load_data(config):
    """Load preprocessed data."""
    processed_dir = config['data']['processed_dir']
    
    features = np.load(os.path.join(processed_dir, 'features.npy'))
    labels = np.load(os.path.join(processed_dir, 'labels.npy'))
    
    # Add channel dimension for CNN (N, H, W) -> (N, 1, H, W)
    features = np.expand_dims(features, axis=1)
    
    return features, labels


def split_data(features, labels, config):
    """Split data into train/val/test sets."""
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    train_split = int(n_samples * config['data']['train_split'])
    val_split = int(n_samples * (config['data']['train_split'] + config['data']['val_split']))
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]
    
    return (features[train_idx], labels[train_idx],
            features[val_idx], labels[val_idx],
            features[test_idx], labels[test_idx])


def create_dataloaders(train_data, val_data, batch_size):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data[0]),
        torch.LongTensor(train_data[1])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data[0]),
        torch.LongTensor(val_data[1])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def create_model(config, device):
    """Create model based on configuration."""
    architecture = config['model']['architecture']
    num_classes = config['model']['num_classes']
    dropout = config['model']['dropout']
    
    if architecture == 'cnn':
        model = AudioCNN(num_classes=num_classes, dropout=dropout)
    elif architecture == 'improved_cnn':
        model = ImprovedAudioCNN(num_classes=num_classes, dropout=dropout)
    elif architecture == 'attention_cnn':
        model = AttentionAudioCNN(num_classes=num_classes, dropout=dropout)
    elif architecture == 'lstm':
        model = AudioLSTM(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    
    return model.to(device)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Train audio classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger('training')
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Set seed
    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    
    # Load and split data
    logger.info('Loading data...')
    features, labels = load_data(config)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(features, labels, config)
    
    logger.info(f'Train samples: {len(X_train)}')
    logger.info(f'Val samples: {len(X_val)}')
    logger.info(f'Test samples: {len(X_test)}')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        (X_train, y_train), (X_val, y_val), 
        config['training']['batch_size']
    )
    
    # Create model
    logger.info(f'Creating {config["model"]["architecture"]} model...')
    model = create_model(config, device)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    # Train
    logger.info('Starting training...')
    trainer = Trainer(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        config['training']['early_stopping']
    )
    trainer.train(config['training']['epochs'])
    
    logger.info('Training complete!')


if __name__ == '__main__':
    main()
