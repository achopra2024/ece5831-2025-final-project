"""
Preprocess ESC-50 dataset for training.
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
from src.features.extractor import extract_mel_spectrogram, extract_mfcc
from src.utils.config import load_config
from src.utils.logger import setup_logger


def preprocess_dataset(config):
    """
    Preprocess entire dataset.
    
    Args:
        config (dict): Configuration dictionary
    """
    logger = setup_logger('preprocessing')
    logger.info('Starting data preprocessing...')
    
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
            
            # Extract features
            if config['features']['type'] == 'mel_spectrogram':
                features = extract_mel_spectrogram(
                    audio,
                    sample_rate=config['data']['sample_rate'],
                    n_mels=config['features']['n_mels'],
                    n_fft=config['features']['n_fft'],
                    hop_length=config['features']['hop_length']
                )
            elif config['features']['type'] == 'mfcc':
                features = extract_mfcc(
                    audio,
                    sample_rate=config['data']['sample_rate'],
                    n_mfcc=config['features']['n_mfcc'],
                    n_fft=config['features']['n_fft'],
                    hop_length=config['features']['hop_length']
                )
            else:
                features = audio
            
            features_list.append(features)
            # Use animal_label if filtering, otherwise use original target
            label = row.get('animal_label', row['target'])
            labels_list.append(label)
            
        except Exception as e:
            logger.error(f'Error processing {row["filename"]}: {str(e)}')
            continue
    
    # Save processed features
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    np.save(os.path.join(processed_dir, 'features.npy'), features_array)
    np.save(os.path.join(processed_dir, 'labels.npy'), labels_array)
    
    logger.info(f'Preprocessing complete!')
    logger.info(f'Features shape: {features_array.shape}')
    logger.info(f'Labels shape: {labels_array.shape}')
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
