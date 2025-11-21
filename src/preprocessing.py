"""
Audio Preprocessing and Feature Extraction Module

This module handles:
- Audio file loading and validation
- Feature extraction (MFCC, Mel-spectrogram, etc.)
- Data augmentation
- Audio normalization
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing and feature extraction class"""

    def __init__(
        self,
        sample_rate: int = 22050,
        duration: float = 4.0,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize audio preprocessor

        Args:
            sample_rate: Target sample rate for audio
            duration: Duration to pad/trim audio files to
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = int(sample_rate * duration)

    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sample rate

        Args:
            file_path: Path to audio file

        Returns:
            Audio time series as numpy array
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {file_path} (duration: {len(audio)/sr:.2f}s)")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise

    def pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or trim audio to fixed length

        Args:
            audio: Audio time series

        Returns:
            Fixed-length audio array
        """
        if len(audio) < self.max_length:
            # Pad with zeros
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        else:
            # Trim to max length
            audio = audio[:self.max_length]
        return audio

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio

        Args:
            audio: Audio time series

        Returns:
            MFCC features (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features

        Args:
            audio: Audio time series

        Returns:
            Mel-spectrogram (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_features(self, audio: np.ndarray, feature_type: str = 'mfcc') -> np.ndarray:
        """
        Extract features from audio based on type

        Args:
            audio: Audio time series
            feature_type: Type of features ('mfcc' or 'mel')

        Returns:
            Feature array
        """
        if feature_type == 'mfcc':
            features = self.extract_mfcc(audio)
        elif feature_type == 'mel':
            features = self.extract_mel_spectrogram(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        return features

    def augment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation techniques

        Args:
            audio: Audio time series

        Returns:
            List of augmented audio arrays
        """
        augmented = [audio]  # Original

        # Time stretching
        audio_stretched = librosa.effects.time_stretch(audio, rate=1.1)
        augmented.append(audio_stretched)

        # Pitch shifting
        audio_pitched = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=2
        )
        augmented.append(audio_pitched)

        # Add white noise
        noise = np.random.randn(len(audio)) * 0.005
        audio_noisy = audio + noise
        augmented.append(audio_noisy)

        return augmented

    def preprocess_file(
        self,
        file_path: str,
        feature_type: str = 'mfcc',
        augment: bool = False
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Complete preprocessing pipeline for a single audio file

        Args:
            file_path: Path to audio file
            feature_type: Type of features to extract
            augment: Whether to apply data augmentation

        Returns:
            Tuple of (features, augmented_features)
        """
        # Load audio
        audio = self.load_audio(file_path)

        # Pad or trim
        audio = self.pad_or_trim(audio)

        # Extract features
        features = self.extract_features(audio, feature_type)

        # Augmentation if requested
        augmented_features = None
        if augment:
            augmented_audio = self.augment_audio(audio)
            augmented_features = [
                self.extract_features(aug_audio, feature_type)
                for aug_audio in augmented_audio
            ]

        return features, augmented_features

    def preprocess_directory(
        self,
        directory: str,
        feature_type: str = 'mfcc'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess all audio files in a directory

        Args:
            directory: Path to directory containing audio files
            feature_type: Type of features to extract

        Returns:
            Tuple of (features_array, file_paths)
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
        features_list = []
        file_paths = []

        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in audio_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        features, _ = self.preprocess_file(file_path, feature_type)
                        features_list.append(features)
                        file_paths.append(file_path)
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {str(e)}")

        features_array = np.array(features_list)
        logger.info(f"Processed {len(features_list)} files from {directory}")

        return features_array, file_paths


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features using standardization

    Args:
        features: Feature array

    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    normalized = (features - mean) / (std + 1e-8)
    return normalized


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioPreprocessor()

    # Example: Process a single file
    # features, aug_features = preprocessor.preprocess_file(
    #     "path/to/audio.wav",
    #     feature_type='mfcc',
    #     augment=True
    # )

    print("Audio preprocessing module loaded successfully!")
    print(f"Sample rate: {preprocessor.sample_rate} Hz")
    print(f"Duration: {preprocessor.duration} seconds")
    print(f"MFCC coefficients: {preprocessor.n_mfcc}")
