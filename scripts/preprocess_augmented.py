"""
Enhanced preprocessing with data augmentation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import load_audio, normalize_audio, pad_audio
from src.data.augmentation import time_shift, pitch_shift, add_noise
from src.features.extractor import extract_mel_spectrogram, extract_mfcc
from src.utils.config import load_config
from src.utils.logger import setup_logger


def augment_audio(audio, config):
    """Apply data augmentation to audio."""
    augmented = [audio]  # Original
    
    aug_config = config['data']['augmentation']
    
    if aug_config.get('time_shift', False):
        augmented.append(time_shift(audio, shift_max=0.2))
    
    if aug_config.get('pitch_shift', False):
        augmented.append(pitch_shift(audio, sample_rate=config['data']['sample_rate'], n_steps=2))
        augmented.append(pitch_shift(audio, sample_rate=config['data']['sample_rate'], n_steps=-2))
    
    if aug_config.get('add_noise', False):
        augmented.append(add_noise(audio, noise_factor=0.005))
    
    if aug_config.get('time_stretch', False):
        import librosa
        augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
        augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    
    return augmented


def preprocess_dataset(config):
    """
    Preprocess entire dataset with augmentation.
    
    Args:
        config (dict): Configuration dictionary
    """
    logger = setup_logger('preprocessing')
    logger.info('Starting data preprocessing with augmentation...')
    
    # Load metadata
    metadata_path = os.path.join(config['data']['data_dir'], 'meta', 'esc50.csv')
    if not os.path.exists(metadata_path):
        logger.error(f'Metadata file not found: {metadata_path}')
        return
    
    metadata = pd.read_csv(metadata_path)
    logger.info(f'Loaded {len(metadata)} audio files from metadata')
    
    # Filter for animal sounds only if configured
    if config['data'].get('filter_animal_sounds', False):
        animal_categories = config['data'].get('animal_categories', [])
        metadata = metadata[metadata['category'].isin(animal_categories)]
        logger.info(f'Filtered to {len(metadata)} animal sound files')
        logger.info(f'Animal categories: {animal_categories}')
        
        # Remap labels to 0-9 for animal classes only
        category_to_label = {cat: idx for idx, cat in enumerate(sorted(animal_categories))}
        metadata['animal_label'] = metadata['category'].map(category_to_label)
        logger.info(f'Label mapping: {category_to_label}')
    
    # Create output directory
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process each audio file
    features_list = []
    labels_list = []
    
    use_augmentation = config['data']['augmentation'].get('enabled', False)
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc='Processing'):
        try:
            # Load audio
            audio_path = os.path.join(
                config['data']['data_dir'], 
                'audio', 
                row['filename']
            )
            
            if not os.path.exists(audio_path):
                logger.warning(f'Audio file not found: {audio_path}')
                continue
            
            audio = load_audio(
                audio_path, 
                sample_rate=config['data']['sample_rate'],
                duration=config['data']['duration']
            )
            
            # Normalize
            audio = normalize_audio(audio)
            
            # Pad to fixed length
            target_length = config['data']['sample_rate'] * config['data']['duration']
            audio = pad_audio(audio, target_length)
            
            # Get label
            label = row.get('animal_label', row['target'])
            
            # Apply augmentation if enabled
            if use_augmentation:
                audio_variants = augment_audio(audio, config)
            else:
                audio_variants = [audio]
            
            # Extract features for each variant
            for audio_variant in audio_variants:
                # Ensure audio is correct length after augmentation
                audio_variant = pad_audio(audio_variant, target_length)
                
                if config['features']['type'] == 'mel_spectrogram':
                    features = extract_mel_spectrogram(
                        audio_variant,
                        sample_rate=config['data']['sample_rate'],
                        n_mels=config['features']['n_mels'],
                        n_fft=config['features']['n_fft'],
                        hop_length=config['features']['hop_length']
                    )
                elif config['features']['type'] == 'mfcc':
                    features = extract_mfcc(
                        audio_variant,
                        sample_rate=config['data']['sample_rate'],
                        n_mfcc=config['features']['n_mfcc'],
                        n_fft=config['features']['n_fft'],
                        hop_length=config['features']['hop_length']
                    )
                else:
                    features = audio_variant
                
                features_list.append(features)
                labels_list.append(label)
            
        except Exception as e:
            logger.error(f'Error processing {row["filename"]}: {str(e)}')
            continue
    
    # Convert to numpy arrays with consistent shapes
    # Stack features to ensure uniform shape
    try:
        features_array = np.stack(features_list)
    except ValueError:
        # If stacking fails, pad/crop to same shape
        logger.warning("Inconsistent feature shapes detected, normalizing...")
        max_shape = max([f.shape for f in features_list], key=lambda x: x[0] * x[1])
        features_normalized = []
        for feat in features_list:
            if feat.shape != max_shape:
                # Pad or crop to match max_shape
                padded = np.zeros(max_shape)
                padded[:feat.shape[0], :feat.shape[1]] = feat
                features_normalized.append(padded)
            else:
                features_normalized.append(feat)
        features_array = np.stack(features_normalized)
    
    labels_array = np.array(labels_list)
    
    np.save(os.path.join(processed_dir, 'features.npy'), features_array)
    np.save(os.path.join(processed_dir, 'labels.npy'), labels_array)
    
    logger.info(f'Preprocessing complete!')
    logger.info(f'Features shape: {features_array.shape}')
    logger.info(f'Labels shape: {labels_array.shape}')
    if use_augmentation:
        logger.info(f'Augmentation: {len(features_list) / len(metadata):.1f}x data increase')
    logger.info(f'Saved to: {processed_dir}')


def main():
    parser = argparse.ArgumentParser(description='Preprocess ESC-50 dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Preprocess dataset
    preprocess_dataset(config)


if __name__ == '__main__':
    main()
