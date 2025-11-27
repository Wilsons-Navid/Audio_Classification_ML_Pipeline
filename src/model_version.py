"""
Model Versioning System

Manages multiple model versions with rollback capability
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manages model versions and provides rollback functionality"""

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.versions_dir = os.path.join(models_dir, 'versions')
        self.registry_file = os.path.join(models_dir, 'model_registry.json')

        # Create directories
        os.makedirs(self.versions_dir, exist_ok=True)

        # Initialize registry
        self.registry = self._load_registry()

    def _load_registry(self):
        """Load model registry from disk"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
                # Normalize paths for cross-platform compatibility
                for version in registry.get('versions', []):
                    if 'model_file' in version:
                        version['model_file'] = version['model_file'].replace('\\', '/')
                    if 'original_model' in version:
                        version['original_model'] = version['original_model'].replace('\\', '/')
                
                # Normalize metadata paths
                for v_id, meta in registry.get('metadata', {}).items():
                    if 'model_file' in meta:
                        meta['model_file'] = meta['model_file'].replace('\\', '/')
                    if 'original_model' in meta:
                        meta['original_model'] = meta['original_model'].replace('\\', '/')
                        
                return registry
        else:
            registry = {
                'current_version': None,
                'versions': [],
                'metadata': {}
            }
            
            # Bootstrap with existing model if available
            main_model_path = os.path.join(self.models_dir, 'vishing_detector_keras3.keras')
            if os.path.exists(main_model_path):
                # Create initial version
                version_id = 'pretrained-model'
                version_dir = os.path.join(self.versions_dir, version_id)
                os.makedirs(version_dir, exist_ok=True)
                
                # Copy model
                shutil.copy2(main_model_path, os.path.join(version_dir, 'model.keras'))
                
                # Create metadata
                version_metadata = {
                    'version_id': version_id,
                    'created_at': datetime.fromtimestamp(os.path.getctime(main_model_path)).isoformat(),
                    'model_file': os.path.join(version_dir, 'model.keras'),
                    'original_model': main_model_path,
                    'metrics': {'accuracy': 'Original'},
                    'status': 'active'
                }
                
                registry['versions'].append(version_metadata)
                registry['metadata'][version_id] = version_metadata
                registry['current_version'] = version_id
                
                # Save immediately
                with open(self.registry_file, 'w') as f:
                    json.dump(registry, f, indent=2)
                    
            return registry

    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def create_version(self, model_path, metadata=None):
        """
        Create a new model version

        Args:
            model_path: Path to the model file to version
            metadata: Dict with model info (accuracy, metrics, etc.)

        Returns:
            version_id (str)
        """
        # Generate version ID
        # Format: retrained_v{count}_acc{accuracy}_{timestamp}
        count = len(self.registry['versions']) + 1
        
        acc_str = ""
        if metadata and 'accuracy' in metadata:
            try:
                acc = float(metadata['accuracy'])
                acc_str = f"_acc{int(acc*100)}"
            except (ValueError, TypeError):
                pass
                
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version_id = f"retrained_v{count}{acc_str}_{timestamp}"

        # Create version directory
        version_dir = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)

        # Copy model file
        model_ext = Path(model_path).suffix
        versioned_model_path = os.path.join(version_dir, f'model{model_ext}').replace('\\', '/')
        shutil.copy2(model_path, versioned_model_path)

        # Copy metadata if it exists
        metadata_path = model_path.replace(model_ext, '_metadata.json')
        if os.path.exists(metadata_path):
            shutil.copy2(metadata_path, os.path.join(version_dir, 'metadata.json'))

        # Create version metadata
        version_metadata = {
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'model_file': versioned_model_path,
            'original_model': model_path,
            'metrics': metadata or {},
            'status': 'active'
        }

        # Update registry
        self.registry['versions'].append(version_metadata)
        self.registry['metadata'][version_id] = version_metadata
        self.registry['current_version'] = version_id

        self._save_registry()

        logger.info(f"Created model version: {version_id}")
        return version_id

    def get_version(self, version_id):
        """Get metadata for a specific version"""
        self.registry = self._load_registry()
        return self.registry['metadata'].get(version_id)

    def list_versions(self, limit=10):
        """List recent model versions"""
        self.registry = self._load_registry()
        versions = sorted(
            self.registry['versions'],
            key=lambda x: x['created_at'],
            reverse=True
        )
        return versions[:limit]

    def get_current_version(self):
        """Get current active version"""
        self.registry = self._load_registry()
        version_id = self.registry.get('current_version')
        if version_id:
            return self.registry['metadata'].get(version_id)
        return None

    def rollback(self, version_id):
        """
        Rollback to a previous model version

        Args:
            version_id: Version to rollback to

        Returns:
            (success, message)
        """
        version = self.get_version(version_id)

        if not version:
            return False, f"Version {version_id} not found"

        if not os.path.exists(version['model_file']):
            return False, f"Model file not found for version {version_id}"

        try:
            # Get original model path
            original_path = version['original_model']
            original_ext = Path(original_path).suffix

            # Backup current model
            if os.path.exists(original_path):
                backup_path = original_path.replace(original_ext, f'_backup{original_ext}')
                shutil.copy2(original_path, backup_path)
                logger.info(f"Backed up current model to {backup_path}")

            # Restore version
            shutil.copy2(version['model_file'], original_path)

            # Restore metadata
            version_metadata_path = os.path.join(
                os.path.dirname(version['model_file']),
                'metadata.json'
            )
            if os.path.exists(version_metadata_path):
                original_metadata = original_path.replace(original_ext, '_metadata.json')
                shutil.copy2(version_metadata_path, original_metadata)

            # Update current version
            self.registry['current_version'] = version_id
            self._save_registry()

            logger.info(f"Successfully rolled back to version {version_id}")
            return True, f"Rolled back to version {version_id}"

        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False, f"Rollback failed: {str(e)}"

    def delete_version(self, version_id):
        """
        Delete a model version

        Args:
            version_id: Version to delete

        Returns:
            (success, message)
        """
        # Don't allow deleting current version
        if version_id == self.registry.get('current_version'):
            return False, "Cannot delete current active version"

        version = self.get_version(version_id)
        if not version:
            return False, f"Version {version_id} not found"

        try:
            # Delete version directory
            version_dir = os.path.dirname(version['model_file'])
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)

            # Remove from registry
            self.registry['versions'] = [
                v for v in self.registry['versions']
                if v['version_id'] != version_id
            ]
            if version_id in self.registry['metadata']:
                del self.registry['metadata'][version_id]

            self._save_registry()

            logger.info(f"Deleted version {version_id}")
            return True, f"Deleted version {version_id}"

        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            return False, f"Delete failed: {str(e)}"

    def compare_versions(self, version_id1, version_id2):
        """Compare metrics between two versions"""
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)

        if not v1 or not v2:
            return None

        comparison = {
            'version_1': {
                'id': version_id1,
                'created': v1['created_at'],
                'metrics': v1.get('metrics', {})
            },
            'version_2': {
                'id': version_id2,
                'created': v2['created_at'],
                'metrics': v2.get('metrics', {})
            },
            'differences': {}
        }

        # Calculate metric differences
        metrics1 = v1.get('metrics', {})
        metrics2 = v2.get('metrics', {})

        for key in set(list(metrics1.keys()) + list(metrics2.keys())):
            if key in metrics1 and key in metrics2:
                try:
                    diff = float(metrics2[key]) - float(metrics1[key])
                    comparison['differences'][key] = {
                        'v1': metrics1[key],
                        'v2': metrics2[key],
                        'change': diff,
                        'improvement': diff > 0
                    }
                except (ValueError, TypeError):
                    pass

        return comparison
