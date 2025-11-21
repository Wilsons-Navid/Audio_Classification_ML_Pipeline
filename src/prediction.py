"""
Prediction Module

This module handles:
- Single audio file prediction
- Batch prediction
- Prediction result formatting
"""

import numpy as np
from typing import Dict, List, Union
import logging

from src.preprocessing import AudioPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPredictor:
    """Audio prediction class"""

    def __init__(self, model, preprocessor: AudioPreprocessor, class_names: List[str]):
        """
        Initialize predictor

        Args:
            model: Trained Keras model
            preprocessor: AudioPreprocessor instance
            class_names: List of class names
        """
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names

    def predict_single(
        self,
        file_path: str,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict class for a single audio file

        Args:
            file_path: Path to audio file
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio
            features, _ = self.preprocessor.preprocess_file(file_path)

            # Reshape for model
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            features = np.expand_dims(features, axis=-1)  # Add channel dimension

            # Predict
            predictions = self.model.predict(features, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])

            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'file_path': file_path
            }

            if return_probabilities:
                probabilities = {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
                result['probabilities'] = probabilities

            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
            return result

        except Exception as e:
            logger.error(f"Error predicting {file_path}: {str(e)}")
            raise

    def predict_batch(
        self,
        file_paths: List[str],
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Predict classes for multiple audio files

        Args:
            file_paths: List of paths to audio files
            return_probabilities: Whether to return class probabilities

        Returns:
            List of prediction result dictionaries
        """
        results = []

        for file_path in file_paths:
            try:
                result = self.predict_single(file_path, return_probabilities)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {file_path}: {str(e)}")
                results.append({
                    'file_path': file_path,
                    'error': str(e)
                })

        logger.info(f"Completed batch prediction for {len(file_paths)} files")
        return results

    def get_top_k_predictions(
        self,
        file_path: str,
        k: int = 3
    ) -> List[Dict]:
        """
        Get top K predictions for an audio file

        Args:
            file_path: Path to audio file
            k: Number of top predictions to return

        Returns:
            List of top K predictions with class and confidence
        """
        result = self.predict_single(file_path, return_probabilities=True)
        probabilities = result['probabilities']

        # Sort by probability
        sorted_predictions = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_k = [
            {'class': class_name, 'confidence': conf}
            for class_name, conf in sorted_predictions[:k]
        ]

        return top_k


if __name__ == "__main__":
    print("Prediction module loaded successfully!")
