"""
Production-Grade API Endpoints
Includes validation, versioning, and better error handling
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import shutil
import logging

from src.model_version import ModelVersionManager
from src.validation import AudioValidator
from src.retrain import get_retrainer

logger = logging.getLogger(__name__)

# Create blueprint
api_v2 = Blueprint('api_v2', __name__)

# Initialize managers
version_manager = ModelVersionManager()
validator = AudioValidator()


def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


@api_v2.route('/validate', methods=['POST'])
def validate_audio():
    """Validate uploaded audio file without saving"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save temporarily
        temp_path = os.path.join('data/uploaded/temp', secure_filename(file.filename))
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)

        # Validate
        is_valid, message = validator.validate_file(temp_path)

        # Get audio info if valid
        audio_info = None
        if is_valid:
            audio_info = validator.get_audio_info(temp_path)

        # Clean up
        os.remove(temp_path)

        return jsonify({
            'valid': is_valid,
            'message': message,
            'audio_info': audio_info
        }), 200

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_v2.route('/versions', methods=['GET'])
def list_versions():
    """List all model versions"""
    try:
        limit = int(request.args.get('limit', 10))
        versions = version_manager.list_versions(limit=limit)

        return jsonify({
            'versions': versions,
            'current_version': version_manager.get_current_version()
        }), 200

    except Exception as e:
        logger.error(f"Error listing versions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_v2.route('/versions/<version_id>', methods=['GET'])
def get_version(version_id):
    """Get details of a specific version"""
    try:
        version = version_manager.get_version(version_id)

        if not version:
            return jsonify({'error': f'Version {version_id} not found'}), 404

        return jsonify(version), 200

    except Exception as e:
        logger.error(f"Error getting version: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_v2.route('/versions/<version_id>/rollback', methods=['POST'])
def rollback_version(version_id):
    """Rollback to a specific model version"""
    try:
        success, message = version_manager.rollback(version_id)

        if success:
            return jsonify({
                'success': True,
                'message': message,
                'version_id': version_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400

    except Exception as e:
        logger.error(f"Rollback error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_v2.route('/versions/compare', methods=['GET'])
def compare_versions():
    """Compare two model versions"""
    try:
        v1 = request.args.get('v1')
        v2 = request.args.get('v2')

        if not v1 or not v2:
            return jsonify({'error': 'Please provide v1 and v2 parameters'}), 400

        comparison = version_manager.compare_versions(v1, v2)

        if not comparison:
            return jsonify({'error': 'One or both versions not found'}), 404

        return jsonify(comparison), 200

    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_v2.route('/upload_with_label', methods=['POST'])
def upload_with_label():
    """Upload training data with user-specified class label and validation"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    class_label = request.form.get('class_label', 'Suspicious')

    # Validate class label
    allowed_labels = ['Legitimate', 'Suspicious']
    if class_label not in allowed_labels:
        return jsonify({
            'error': f'Invalid class label. Allowed: {", ".join(allowed_labels)}'
        }), 400

    try:
        class_dir = os.path.join('data/uploaded/training', class_label)
        os.makedirs(class_dir, exist_ok=True)

        saved_files = []
        invalid_files = []
        temp_dir = os.path.join('data/uploaded', 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Process each file
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(temp_dir, filename)

                # Save temporarily
                file.save(temp_path)

                # Validate
                is_valid, error_msg = validator.validate_file(temp_path)

                if is_valid:
                    # Move to class directory
                    final_path = os.path.join(class_dir, filename)
                    shutil.move(temp_path, final_path)
                    saved_files.append(filename)
                    logger.info(f"✓ Saved valid file: {filename}")
                else:
                    # Remove invalid file
                    os.remove(temp_path)
                    invalid_files.append({'file': filename, 'error': error_msg})
                    logger.warning(f"✗ Rejected invalid file: {filename} - {error_msg}")

        # Clean up temp directory
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

        response = {
            'message': f'Uploaded {len(saved_files)} valid files to class "{class_label}"',
            'class_label': class_label,
            'valid_files': saved_files,
            'valid_count': len(saved_files),
            'invalid_count': len(invalid_files)
        }

        if invalid_files:
            response['invalid_files'] = invalid_files
            response['warning'] = f'{len(invalid_files)} files failed validation'

        status_code = 200 if saved_files else 400
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
