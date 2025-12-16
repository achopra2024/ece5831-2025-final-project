"""
Audio visualization functions.
"""

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def plot_waveform(audio, sample_rate=44100, title='Waveform'):
    """
    Plot audio waveform.
    
    Args:
        audio (np.ndarray): Audio time series
        sample_rate (int): Sample rate
        title (str): Plot title
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def plot_spectrogram(audio, sample_rate=44100, title='Spectrogram'):
    """
    Plot spectrogram of audio.
    
    Args:
        audio (np.ndarray): Audio time series
        sample_rate (int): Sample rate
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram(mel_spec, sample_rate=44100, title='Mel Spectrogram'):
    """
    Plot Mel spectrogram.
    
    Args:
        mel_spec (np.ndarray): Mel spectrogram
        sample_rate (int): Sample rate
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_spec, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mfcc(mfcc, sample_rate=44100, title='MFCC'):
    """
    Plot MFCC features.
    
    Args:
        mfcc (np.ndarray): MFCC features
        sample_rate (int): Sample rate
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()
