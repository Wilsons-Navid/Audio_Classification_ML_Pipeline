"""
Model Retraining Module

Handles retraining of the audio classification model with new data
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import AudioPreprocessor
from src.model import AudioClassifier
from src.utils import save_json
from src.model_version import ModelVersionManager
from src.validation import AudioValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """Handles model retraining process"""

    def __init__(self, data_dir='data', model_dir='models', logs_dir='logs'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.logs_dir = logs_dir
        self.preprocessor = AudioPreprocessor()
        self.version_manager = ModelVersionManager(model_dir)
        self.validator = AudioValidator()
        self.status = {
            'status': 'idle',
            'progress': 0,
            'message': '',
            'started_at': None,
            'completed_at': None,
            'error': None,
            'metrics': {},
            'version_id': None
        }

    def get_status(self):
        """Get current retraining status"""
        return self.status

    def update_status(self, status, progress=None, message='', error=None, metrics=None):
        """Update retraining status"""
        self.status['status'] = status
        if progress is not None:
            self.status['progress'] = progress
        if message:
            self.status['message'] = message
        if error:
            self.status['error'] = error
        if metrics:
            self.status['metrics'] = metrics

        # Save status to file
        status_file = os.path.join(self.logs_dir, 'retrain_status.json')
        save_json(self.status, status_file)
        logger.info(f"Status: {status} - {message} ({progress}%)")

    def organize_ravdess_data(self):
        """Organize RAVDESS data from raw directory"""
        raw_ravdess = os.path.join(self.data_dir, 'raw', 'RAVDESS')

        if not os.path.exists(raw_ravdess):
            logger.warning("RAVDESS raw data not found")
            return

        logger.info("Organizing RAVDESS data...")

        # Emotion mapping (for binary classification)
        # 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        # Legitimate: neutral, calm, happy
        # Suspicious: angry, fearful, sad, disgust, surprised

        legitimate_emotions = ['01', '02', '03']  # neutral, calm, happy
        suspicious_emotions = ['04', '05', '06', '07', '08']  # sad, angry, fearful, disgust, surprised

        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        # Create class directories
        for split_dir in [train_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, 'Legitimate'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'Suspicious'), exist_ok=True)

        # Process each actor directory
        actor_dirs = [d for d in os.listdir(raw_ravdess) if d.startswith('Actor_')]

        import random
        random.seed(42)

        files_organized = {'train': 0, 'test': 0}

        for actor_dir in actor_dirs:
            actor_path = os.path.join(raw_ravdess, actor_dir)
            audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

            for audio_file in audio_files:
                # Parse filename: 03-01-06-01-02-01-12.wav
                # Position 3 (index 2) is emotion
                parts = audio_file.split('-')
                if len(parts) >= 3:
                    emotion = parts[2]

                    # Determine class
                    if emotion in legitimate_emotions:
                        class_name = 'Legitimate'
                    elif emotion in suspicious_emotions:
                        class_name = 'Suspicious'
                    else:
                        continue

                    # 80/20 train/test split
                    split = 'train' if random.random() < 0.8 else 'test'

                    # Copy file
                    src = os.path.join(actor_path, audio_file)
                    dst_dir = os.path.join(train_dir if split == 'train' else test_dir, class_name)
                    dst = os.path.join(dst_dir, audio_file)

                    shutil.copy2(src, dst)
                    files_organized[split] += 1

        logger.info(f"Organized {files_organized['train']} training and {files_organized['test']} test files")

    def load_training_data(self):
        """Load data ONLY from uploaded files for fast retraining (Transfer Learning)"""
        uploaded_dir = os.path.join(self.data_dir, 'uploaded', 'training')
        
        # Hardcode class names to match pretrained model and ensure consistency
        class_names = ['Legitimate', 'Suspicious']
        all_files = {c: [] for c in class_names}
        
        logger.info("FAST RETRAINING MODE: Skipping original dataset, loading only uploaded files...")

        # Add uploaded files
        if os.path.exists(uploaded_dir):
            uploaded_classes = [d for d in os.listdir(uploaded_dir) 
                               if os.path.isdir(os.path.join(uploaded_dir, d))]
            
            for uploaded_class in uploaded_classes:
                # Map uploaded class (could make this configurable)
                mapped_class = 'Suspicious' if uploaded_class == 'new_class' else uploaded_class
                
                if mapped_class in all_files:
                    uploaded_class_dir = os.path.join(uploaded_dir, uploaded_class)
                    audio_files = [os.path.join(uploaded_class_dir, f) 
                                  for f in os.listdir(uploaded_class_dir) 
                                  if f.endswith('.wav')]
                    all_files[mapped_class].extend(audio_files)
                    logger.info(f"Added {len(audio_files)} uploaded files to class '{mapped_class}'")
                else:
                    logger.warning(f"Uploaded class '{uploaded_class}' mapped to '{mapped_class}' not in target classes {class_names}")

        X = []
        y = []
        
        for class_idx, class_name in enumerate(class_names):
            files = all_files.get(class_name, [])
            logger.info(f"Loading {len(files)} files from class '{class_name}'")
            
            for file_path in files:
                try:
                    features, _ = self.preprocessor.preprocess_file(file_path)
                    X.append(features)
                    y.append(class_idx)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")

        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
             logger.warning("No uploaded data found! Training will likely fail or do nothing.")
        else:
            # Add channel dimension
            X = X[..., np.newaxis]

        logger.info(f"Loaded {len(X)} samples for retraining, shape: {X.shape}")
        
        return X, y, class_names

    def load_test_data(self):
        """Load data from test directory"""
        test_dir = os.path.join(self.data_dir, 'test')

        if not os.path.exists(test_dir):
            logger.warning("Test directory not found, skipping test evaluation")
            return None, None, None

        class_dirs = [d for d in os.listdir(test_dir)
                     if os.path.isdir(os.path.join(test_dir, d))]

        X_test = []
        y_test = []
        class_names = sorted(class_dirs)

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(test_dir, class_name)
            audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

            for audio_file in audio_files:
                file_path = os.path.join(class_dir, audio_file)
                try:
                    features, _ = self.preprocessor.preprocess_file(file_path)
                    X_test.append(features)
                    y_test.append(class_idx)
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {str(e)}")

        if not X_test:
            return None, None, None

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_test = X_test[..., np.newaxis]

        logger.info(f"Loaded {len(X_test)} test samples")

        return X_test, y_test, class_names

    def retrain(self, epochs=30, batch_size=32):
        """
        Retrain the model with existing and new data

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            self.update_status('running', 0, 'Starting retraining process...')
            self.status['started_at'] = datetime.now().isoformat()

            # Step 1: Load training data (10%)
            self.update_status('running', 10, 'Loading training data...')
            X_train, y_train, class_names = self.load_training_data()

            # Step 2: Load test data (20%)
            self.update_status('running', 20, 'Loading test data...')
            X_test, y_test, _ = self.load_test_data()

            # Step 3: Create or Load model (30%)
            self.update_status('running', 30, 'Loading/Building model architecture...')
            input_shape = X_train.shape[1:3]  # (height, width)
            num_classes = len(class_names)

            classifier = AudioClassifier(
                input_shape=input_shape,
                num_classes=num_classes,
                model_type='cnn'
            )
            classifier.class_names = class_names

            # Try to load existing model weights for fine-tuning (Transfer Learning)
            main_model_path = os.path.join(self.model_dir, 'vishing_detector_keras3.keras')
            if os.path.exists(main_model_path):
                try:
                    logger.info(f"Loading existing model from {main_model_path} for fine-tuning...")
                    # Load weights into the new model
                    # We use load_model from the class but need to handle the instance correctly
                    loaded_instance = AudioClassifier.load_model(main_model_path)
                    classifier.model = loaded_instance.model
                    logger.info("Successfully loaded existing model weights.")
                except Exception as e:
                    logger.warning(f"Could not load existing model for fine-tuning: {e}. Starting from scratch.")
                    classifier.build_model()
            else:
                logger.info("No existing model found. Training from scratch.")
                classifier.build_model()

            # Step 4: Train model (40-80%)
            self.update_status('running', 40, f'Training model on {len(X_train)} samples...')

            history = classifier.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=epochs,
                batch_size=batch_size
            )

            self.update_status('running', 80, 'Training completed, evaluating...')

            # Step 5: Evaluate model (85%)
            self.update_status('running', 85, 'Evaluating model performance...')

            if X_test is not None and y_test is not None:
                metrics = classifier.evaluate(X_test, y_test)
                logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            else:
                metrics = {'accuracy': 'N/A (no test data)'}

            # Step 6: Save model with versioning (90%)
            self.update_status('running', 90, 'Saving retrained model with versioning...')

            # Save model
            main_model_path = os.path.join(self.model_dir, 'vishing_detector_keras3.keras')
            classifier.model.save(main_model_path)
            logger.info(f"Model saved to {main_model_path}")

            # Create model version
            version_metadata = {
                'accuracy': metrics.get('accuracy'),
                'training_samples': len(X_train),
                'test_samples': len(X_test) if X_test is not None else 0,
                'epochs': epochs,
                'batch_size': batch_size,
                'classes': class_names
            }

            version_id = self.version_manager.create_version(
                main_model_path,
                metadata=version_metadata
            )
            logger.info(f"Created model version: {version_id}")
            self.status['version_id'] = version_id

            # Step 7: Save metadata (95%)
            self.update_status('running', 95, 'Saving model metadata...')

            metadata = {
                'timestamp': datetime.now().isoformat(),
                'input_shape': list(input_shape),
                'num_classes': num_classes,
                'class_names': class_names,
                'training_samples': len(X_train),
                'test_samples': len(X_test) if X_test is not None else 0,
                'epochs': epochs,
                'batch_size': batch_size,
                'metrics': metrics
            }

            metadata_path = os.path.join(self.model_dir, 'vishing_detector_metadata.json')
            save_json(metadata, metadata_path)

            # Save training history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join(self.logs_dir, f'training_history_{timestamp}.json')
            history_data = {
                'loss': [float(x) for x in history.history.get('loss', [])],
                'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])] if X_test is not None else [],
                'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])] if X_test is not None else []
            }
            save_json(history_data, history_path)

            # Complete (100%)
            self.status['completed_at'] = datetime.now().isoformat()
            self.update_status(
                'completed',
                100,
                f'Retraining completed successfully! Model saved to {main_model_path}',
                metrics=metadata
            )

            return True, metadata

        except Exception as e:
            error_msg = f"Retraining failed: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()

            self.update_status('failed', self.status['progress'], error_msg, error=str(e))
            return False, {'error': str(e)}


# Singleton instance
_retrainer = None

def get_retrainer():
    """Get or create retrainer instance"""
    global _retrainer
    if _retrainer is None:
        _retrainer = ModelRetrainer()
    return _retrainer
