"""
Generate comprehensive visualizations for the animal sound recognition project.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import load_audio, normalize_audio
from src.features.extractor import (
    extract_mel_spectrogram, extract_mfcc, 
    extract_chroma, extract_spectral_features
)
from src.visualization.audio_plots import (
    plot_waveform, plot_spectrogram, 
    plot_mel_spectrogram, plot_mfcc
)
from src.visualization.results_plots import plot_class_distribution
from src.utils.config import load_config
from src.utils.logger import setup_logger


def create_output_dir(output_dir):
    """Create output directory for visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'waveforms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mel_spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mfcc'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dataset_analysis'), exist_ok=True)


def plot_class_distribution_enhanced(metadata, output_dir, animal_categories):
    """Plot enhanced class distribution."""
    # Filter animal sounds
    animal_data = metadata[metadata['category'].isin(animal_categories)]
    
    # Count by category
    category_counts = animal_data['category'].value_counts().sort_index()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    axes[0].bar(range(len(category_counts)), category_counts.values, color='steelblue')
    axes[0].set_xlabel('Animal Category', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Animal Sound Dataset Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(category_counts)))
    axes[0].set_xticklabels(category_counts.index, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    colors = plt.cm.Set3(range(len(category_counts)))
    axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Category Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis', 'class_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class distribution plot")


def plot_duration_analysis(metadata, config, output_dir):
    """Plot audio duration statistics."""
    data_dir = config['data']['data_dir']
    durations = []
    
    print("Analyzing audio durations...")
    for idx, row in tqdm(metadata.iterrows(), total=min(50, len(metadata)), desc='Loading samples'):
        if idx >= 50:  # Sample first 50 for speed
            break
        audio_path = os.path.join(data_dir, 'audio', row['filename'])
        if os.path.exists(audio_path):
            audio, sr = librosa.load(audio_path, sr=None)
            durations.append(len(audio) / sr)
    
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=20, color='teal', alpha=0.7, edgecolor='black')
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Audio Duration Distribution', fontsize=14, fontweight='bold')
    plt.axvline(np.mean(durations), color='red', linestyle='--', 
                label=f'Mean: {np.mean(durations):.2f}s')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis', 'duration_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved duration analysis")


def visualize_sample_from_each_class(metadata, config, output_dir, animal_categories):
    """Create visualizations for one sample from each animal class."""
    data_dir = config['data']['data_dir']
    sample_rate = config['data']['sample_rate']
    
    # Filter animal sounds
    animal_data = metadata[metadata['category'].isin(animal_categories)]
    
    print("Generating visualizations for each animal class...")
    for category in sorted(animal_categories):
        # Get first sample from this category
        category_samples = animal_data[animal_data['category'] == category]
        if len(category_samples) == 0:
            continue
            
        sample = category_samples.iloc[0]
        audio_path = os.path.join(data_dir, 'audio', sample['filename'])
        
        if not os.path.exists(audio_path):
            continue
        
        # Load audio
        audio = load_audio(audio_path, sample_rate=sample_rate, duration=5)
        audio = normalize_audio(audio)
        
        # 1. Waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.title(f'Waveform - {category.capitalize()}', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'waveforms', f'{category}_waveform.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Spectrogram
        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {category.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spectrograms', f'{category}_spectrogram.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mel Spectrogram
        mel_spec = extract_mel_spectrogram(
            audio, sample_rate=sample_rate,
            n_mels=config['features']['n_mels'],
            n_fft=config['features']['n_fft'],
            hop_length=config['features']['hop_length']
        )
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mel_spec, sr=sample_rate, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {category.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mel_spectrograms', f'{category}_mel_spectrogram.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. MFCC
        mfcc = extract_mfcc(
            audio, sample_rate=sample_rate,
            n_mfcc=config['features']['n_mfcc'],
            n_fft=config['features']['n_fft'],
            hop_length=config['features']['hop_length']
        )
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time', cmap='coolwarm')
        plt.colorbar()
        plt.title(f'MFCC - {category.capitalize()}', fontsize=14, fontweight='bold')
        plt.ylabel('MFCC Coefficients', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mfcc', f'{category}_mfcc.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated visualizations for: {category}")


def create_comparison_grid(metadata, config, output_dir, animal_categories):
    """Create a grid comparing all animals side by side."""
    data_dir = config['data']['data_dir']
    sample_rate = config['data']['sample_rate']
    animal_data = metadata[metadata['category'].isin(animal_categories)]
    
    categories = sorted(animal_categories)[:6]  # First 6 for grid
    n_categories = len(categories)
    
    # Waveform comparison
    fig, axes = plt.subplots(n_categories, 1, figsize=(14, 2*n_categories))
    for idx, category in enumerate(categories):
        sample = animal_data[animal_data['category'] == category].iloc[0]
        audio_path = os.path.join(data_dir, 'audio', sample['filename'])
        if os.path.exists(audio_path):
            audio = load_audio(audio_path, sample_rate=sample_rate, duration=5)
            librosa.display.waveshow(audio, sr=sample_rate, ax=axes[idx])
            axes[idx].set_title(category.capitalize(), fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Amplitude')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis', 'waveform_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved waveform comparison grid")
    
    # Mel spectrogram comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, category in enumerate(categories):
        sample = animal_data[animal_data['category'] == category].iloc[0]
        audio_path = os.path.join(data_dir, 'audio', sample['filename'])
        if os.path.exists(audio_path):
            audio = load_audio(audio_path, sample_rate=sample_rate, duration=5)
            mel_spec = extract_mel_spectrogram(audio, sample_rate=sample_rate)
            img = librosa.display.specshow(mel_spec, sr=sample_rate, x_axis='time', 
                                          y_axis='mel', ax=axes[idx], cmap='magma')
            axes[idx].set_title(category.capitalize(), fontsize=12, fontweight='bold')
            fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis', 'mel_spectrogram_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved mel spectrogram comparison grid")


def plot_feature_statistics(config, output_dir):
    """Plot statistics of extracted features."""
    processed_dir = config['data']['processed_dir']
    
    # Load processed features
    if not os.path.exists(os.path.join(processed_dir, 'features.npy')):
        print("⚠ Processed features not found. Run preprocessing first.")
        return
    
    features = np.load(os.path.join(processed_dir, 'features.npy'))
    labels = np.load(os.path.join(processed_dir, 'labels.npy'))
    
    print(f"Feature shape: {features.shape}")
    
    # Feature statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean feature values per class
    unique_labels = np.unique(labels)
    mean_features = [features[labels == label].mean() for label in unique_labels]
    std_features = [features[labels == label].std() for label in unique_labels]
    
    axes[0, 0].bar(unique_labels, mean_features, color='steelblue')
    axes[0, 0].set_xlabel('Class Label')
    axes[0, 0].set_ylabel('Mean Feature Value')
    axes[0, 0].set_title('Average Feature Values by Class', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    axes[0, 1].bar(unique_labels, std_features, color='coral')
    axes[0, 1].set_xlabel('Class Label')
    axes[0, 1].set_ylabel('Std Feature Value')
    axes[0, 1].set_title('Feature Variability by Class', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Feature distribution
    axes[1, 0].hist(features.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Overall Feature Distribution', fontweight='bold')
    axes[1, 0].set_yscale('log')
    
    # Class balance
    unique, counts = np.unique(labels, return_counts=True)
    axes[1, 1].bar(unique, counts, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Class Label')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Class Distribution in Processed Data', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis', 'feature_statistics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature statistics")


def generate_summary_report(metadata, config, output_dir, animal_categories):
    """Generate a text summary report."""
    animal_data = metadata[metadata['category'].isin(animal_categories)]
    
    report = []
    report.append("=" * 60)
    report.append("ANIMAL SOUND RECOGNITION - DATASET SUMMARY")
    report.append("=" * 60)
    report.append(f"\nTotal ESC-50 samples: {len(metadata)}")
    report.append(f"Animal sound samples: {len(animal_data)}")
    report.append(f"Number of animal categories: {len(animal_categories)}")
    report.append(f"\nAnimal Categories:")
    for cat in sorted(animal_categories):
        count = len(animal_data[animal_data['category'] == cat])
        report.append(f"  - {cat.capitalize()}: {count} samples")
    
    report.append(f"\nConfiguration:")
    report.append(f"  Sample Rate: {config['data']['sample_rate']} Hz")
    report.append(f"  Duration: {config['data']['duration']} seconds")
    report.append(f"  Feature Type: {config['features']['type']}")
    report.append(f"  N Mels: {config['features']['n_mels']}")
    report.append(f"  N MFCC: {config['features']['n_mfcc']}")
    
    report.append(f"\nData Splits:")
    report.append(f"  Train: {config['data']['train_split']*100}%")
    report.append(f"  Validation: {config['data']['val_split']*100}%")
    report.append(f"  Test: {config['data']['test_split']*100}%")
    
    report.append("\n" + "=" * 60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Saved dataset summary report")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset visualizations')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/visualizations',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    output_dir = args.output
    create_output_dir(output_dir)
    
    print("\n" + "="*60)
    print("GENERATING DATASET VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load metadata
    metadata_path = os.path.join(config['data']['data_dir'], 'meta', 'esc50.csv')
    metadata = pd.read_csv(metadata_path)
    animal_categories = config['data']['animal_categories']
    
    # Generate visualizations
    print("\n1. Class Distribution Analysis")
    plot_class_distribution_enhanced(metadata, output_dir, animal_categories)
    
    print("\n2. Duration Analysis")
    plot_duration_analysis(metadata, config, output_dir)
    
    print("\n3. Individual Class Visualizations")
    visualize_sample_from_each_class(metadata, config, output_dir, animal_categories)
    
    print("\n4. Comparison Grids")
    create_comparison_grid(metadata, config, output_dir, animal_categories)
    
    print("\n5. Feature Statistics")
    plot_feature_statistics(config, output_dir)
    
    print("\n6. Summary Report")
    generate_summary_report(metadata, config, output_dir, animal_categories)
    
    print("\n" + "="*60)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
