"""
Model Training and Evaluation Module

This module handles:
- Model architecture definition
- Model training
- Model evaluation
- Model saving/loading
"""

import os
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioClassifier:
    """Audio classification model class"""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        model_type: str = 'cnn'
    ):
        """
        Initialize audio classifier

        Args:
            input_shape: Shape of input features (height, width)
            num_classes: Number of output classes
            model_type: Type of model ('cnn', 'lstm', or 'cnn_lstm')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
        self.class_names = None

    def build_cnn_model(self) -> keras.Model:
        """
        Build CNN model for audio classification

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.input_shape, 1)),

            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_lstm_model(self) -> keras.Model:
        """
        Build LSTM model for audio classification

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),

            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),

            layers.LSTM(64),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_cnn_lstm_model(self) -> keras.Model:
        """
        Build hybrid CNN-LSTM model

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(*self.input_shape, 1)),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Reshape for LSTM
            layers.Reshape((-1, 64)),

            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),

            layers.LSTM(32),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_model(self) -> keras.Model:
        """
        Build model based on specified type

        Returns:
            Compiled Keras model
        """
        if self.model_type == 'cnn':
            model = self.build_cnn_model()
        elif self.model_type == 'lstm':
            model = self.build_lstm_model()
        elif self.model_type == 'cnn_lstm':
            model = self.build_cnn_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info(f"Built {self.model_type} model")
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        class_names: Optional[list] = None
    ) -> keras.callbacks.History:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            class_names: List of class names

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Store class names
        self.class_names = class_names

        # Prepare data
        if self.model_type == 'lstm':
            # LSTM expects (batch, time_steps, features)
            pass  # Already in correct shape
        else:
            # CNN expects (batch, height, width, channels)
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=-1)
                if X_val is not None:
                    X_val = np.expand_dims(X_val, axis=-1)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None

        logger.info("Starting model training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed!")
        return self.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare data
        if self.model_type != 'lstm' and len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, axis=-1)

        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")

        return metrics

    def save_model(self, model_path: str = 'models/audio_classifier.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'class_names': self.class_names,
            'trained_at': datetime.now().isoformat()
        }

        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path: str = 'models/audio_classifier.h5'):
        """Load trained model"""
        # Load metadata
        base_path, _ = os.path.splitext(model_path)
        # Check for both naming conventions
        metadata_path = f"{base_path}_metadata.json"
        
        # Fallback for legacy naming if needed
        if not os.path.exists(metadata_path) and model_path.endswith('.h5'):
             metadata_path = model_path.replace('.h5', '_metadata.json')
             
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            input_shape=tuple(metadata['input_shape']),
            num_classes=metadata['num_classes'],
            model_type=metadata['model_type']
        )

        # Load model
        instance.model = keras.models.load_model(model_path)
        instance.class_names = metadata['class_names']

        logger.info(f"Model loaded from {model_path}")
        return instance


if __name__ == "__main__":
    print("Model module loaded successfully!")
