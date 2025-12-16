"""
Feature extraction functions for audio signals.
"""

import librosa
import numpy as np


def extract_mfcc(audio, sample_rate=44100, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from audio.
    
    Args:
        audio (np.ndarray): Audio time series
        sample_rate (int): Sample rate
        n_mfcc (int): Number of MFCCs to extract
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
    
    Returns:
        np.ndarray: MFCC features
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc


def extract_mel_spectrogram(audio, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract Mel spectrogram from audio.
    
    Args:
        audio (np.ndarray): Audio time series
        sample_rate (int): Sample rate
        n_mels (int): Number of Mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
    
    Returns:
        np.ndarray: Mel spectrogram
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_chroma(audio, sample_rate=44100):
    """
    Extract Chroma features from audio.
    
    Args:
        audio (np.ndarray): Audio time series
        sample_rate (int): Sample rate
    
    Returns:
        np.ndarray: Chroma features
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    return chroma


def extract_spectral_features(audio, sample_rate=44100):
    """
    Extract various spectral features.
    
    Args:
        audio (np.ndarray): Audio time series
        sample_rate (int): Sample rate
    
    Returns:
        dict: Dictionary of spectral features
    """
    features = {
        'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)
    }
    return features
