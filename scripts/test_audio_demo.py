#!/usr/bin/env python3
"""
Audio Demo Test Script for Animal Sound Recognition

This script tests the trained model directly on audio files for quick demos.

Usage:
    python scripts/test_audio_demo.py --audio path/to/audio.wav --model models/saved_models/best_model.pth
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.features.extractor import extract_mel_spectrogram, extract_mfcc
from src.data.preprocessing import normalize_audio, pad_audio, load_audio
from src.models.improved_cnn import ImprovedAudioCNN, AttentionAudioCNN
from src.models.cnn import AudioCNN


class AudioDemo:
    """Demo class for testing trained models on audio files."""
    
    def __init__(self, model_path, config_path=None):
        """
        Initialize the demo.
        
        Args:
            model_path (str): Path to trained model
            config_path (str): Path to config file (optional)
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        if config_path is None:
            config_path = project_root / 'configs' / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Animal categories
        self.categories = self.config['data']['animal_categories']
        
        # Model parameters
        self.sample_rate = self.config['data']['sample_rate']
        self.duration = self.config['data']['duration']
        self.n_mels = self.config['features']['n_mels']
        self.n_fft = self.config['features']['n_fft']
        self.hop_length = self.config['features']['hop_length']
        self.feature_type = self.config['features']['type']
        
        # Load model
        self.model = self._load_model()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {self.device}")
        print(f"✓ Ready to predict animal sounds!\n")
    
    def _load_model(self):
        """Load the trained model."""
        architecture = self.config['model']['architecture']
        num_classes = self.config['model']['num_classes']
        dropout = self.config['model']['dropout']
        
        # Initialize model based on architecture
        if architecture == 'attention_cnn':
            model = AttentionAudioCNN(
                num_classes=num_classes,
                input_channels=1,
                dropout=dropout
            )
        elif architecture == 'improved_cnn':
            model = ImprovedAudioCNN(
                num_classes=num_classes,
                input_channels=1,
                dropout=dropout
            )
        else:
            model = AudioCNN(
                num_classes=num_classes,
                input_channels=1,
                dropout=dropout
            )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_audio(self, audio_path):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            torch.Tensor: Feature tensor
        """
        # Load audio
        audio = load_audio(audio_path, self.sample_rate, self.duration)
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Pad to target length
        target_length = self.sample_rate * self.duration
        audio = pad_audio(audio, target_length)
        
        # Extract features
        if self.feature_type == 'mel_spectrogram':
            features = extract_mel_spectrogram(
                audio,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
        elif self.feature_type == 'mfcc':
            features = extract_mfcc(
                audio,
                sample_rate=self.sample_rate,
                n_mfcc=self.config['features']['n_mfcc'],
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()
        features_tensor = features_tensor.unsqueeze(0).unsqueeze(0)
        
        return features_tensor
    
    def predict(self, audio_path):
        """
        Predict animal sound from audio file.
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            dict: Prediction results
        """
        # Preprocess
        features = self.preprocess_audio(audio_path)
        
        # Predict
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_val = confidence.item()
            all_probs = probabilities.cpu().numpy()[0]
        
        # Prepare results
        results = {
            'audio_path': audio_path,
            'predicted_class': self.categories[predicted_idx],
            'predicted_index': predicted_idx,
            'confidence': confidence_val,
            'all_predictions': [
                {'class': self.categories[i], 'probability': float(prob)}
                for i, prob in enumerate(all_probs)
            ]
        }
        
        return results
    
    def display_results(self, results):
        """Display prediction results."""
        print("\n" + "=" * 70)
        print("🔊 AUDIO DEMO - ANIMAL SOUND PREDICTION")
        print("=" * 70)
        print(f"\nAudio: {os.path.basename(results['audio_path'])}")
        print(f"\n{'PREDICTION':^70}")
        print("-" * 70)
        print(f"  Animal: {results['predicted_class'].upper()}")
        print(f"  Confidence: {results['confidence']*100:.2f}%")
        print("\n" + "=" * 70)
        
        # Sort by probability
        sorted_predictions = sorted(
            results['all_predictions'],
            key=lambda x: x['probability'],
            reverse=True
        )
        
        print(f"\n{'TOP 5 PREDICTIONS':^70}")
        print("-" * 70)
        for i, pred in enumerate(sorted_predictions[:5], 1):
            bar_length = int(pred['probability'] * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {i}. {pred['class']:12s} {bar} {pred['probability']*100:5.2f}%")
        
        print("\n" + "=" * 70 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Test trained animal sound model on audio files'
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/saved_models/best_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)
    
    model_path = project_root / args.model
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    # Initialize and run demo
    demo = AudioDemo(model_path, args.config)
    results = demo.predict(args.audio)
    demo.display_results(results)


if __name__ == '__main__':
    main()
