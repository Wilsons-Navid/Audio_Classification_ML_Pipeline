"""
Audio Data Validation Module

Validates uploaded audio files for quality and format
"""

import os
import librosa
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioValidator:
    """Validates audio files before processing"""

    def __init__(
        self,
        allowed_formats=['.wav', '.mp3', '.flac', '.ogg'],
        min_duration=0.5,  # seconds
        max_duration=30.0,  # seconds
        max_file_size=50 * 1024 * 1024,  # 50MB
        target_sample_rate=22050
    ):
        self.allowed_formats = allowed_formats
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_file_size = max_file_size
        self.target_sample_rate = target_sample_rate

    def validate_file(self, file_path):
        """
        Validate a single audio file

        Returns:
            (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return False, f"File too large: {file_size / 1024 / 1024:.2f}MB (max: {self.max_file_size / 1024 / 1024}MB)"

            if file_size == 0:
                return False, "File is empty"

            # Check file extension
            ext = Path(file_path).suffix.lower()
            if ext not in self.allowed_formats:
                return False, f"Invalid format: {ext}. Allowed: {', '.join(self.allowed_formats)}"

            # Try to load audio
            try:
                audio, sr = librosa.load(file_path, sr=None, duration=self.max_duration + 1)
            except Exception as e:
                return False, f"Cannot load audio file: {str(e)}"

            # Check duration
            duration = len(audio) / sr
            if duration < self.min_duration:
                return False, f"Audio too short: {duration:.2f}s (min: {self.min_duration}s)"

            if duration > self.max_duration:
                return False, f"Audio too long: {duration:.2f}s (max: {self.max_duration}s)"

            # Check for silence (empty audio)
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-4:
                return False, "Audio appears to be silent or corrupted"

            # Check for clipping (audio quality issue)
            clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
            if clipping_ratio > 0.01:
                logger.warning(f"Audio may be clipped: {clipping_ratio*100:.2f}% of samples near max")

            # Check sample rate
            if sr < 8000:
                return False, f"Sample rate too low: {sr}Hz (min: 8000Hz)"

            return True, "Valid"

        except Exception as e:
            logger.error(f"Validation error for {file_path}: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def validate_batch(self, file_paths):
        """
        Validate multiple audio files

        Returns:
            {
                'valid': [list of valid file paths],
                'invalid': [{'path': path, 'error': error_msg}, ...],
                'summary': {'total': N, 'valid': N, 'invalid': N}
            }
        """
        results = {
            'valid': [],
            'invalid': [],
            'summary': {'total': 0, 'valid': 0, 'invalid': 0}
        }

        for file_path in file_paths:
            results['summary']['total'] += 1
            is_valid, message = self.validate_file(file_path)

            if is_valid:
                results['valid'].append(file_path)
                results['summary']['valid'] += 1
            else:
                results['invalid'].append({
                    'path': os.path.basename(file_path),
                    'error': message
                })
                results['summary']['invalid'] += 1

        logger.info(f"Validation: {results['summary']['valid']}/{results['summary']['total']} files valid")

        return results

    def get_audio_info(self, file_path):
        """Get detailed information about an audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr

            return {
                'duration': float(duration),
                'sample_rate': int(sr),
                'channels': 1,  # librosa loads as mono by default
                'samples': len(audio),
                'rms_energy': float(np.sqrt(np.mean(audio**2))),
                'peak_amplitude': float(np.max(np.abs(audio))),
                'file_size_mb': os.path.getsize(file_path) / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return None
