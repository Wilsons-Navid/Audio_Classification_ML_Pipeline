"""
Flask API for Voice Phishing Detection - WORKING VERSION
Fixes Keras compatibility issue
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import shutil
from werkzeug.utils import secure_filename
import numpy as np
import logging
from datetime import datetime
import json
import threading

# Import TensorFlow with compatibility mode
import tensorflow as tf
from tensorflow import keras
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Import retraining and versioning modules
from src.retrain import get_retrainer
from src.model_version import ModelVersionManager
from src.validation import AudioValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            static_folder='../ui/static',
            template_folder='../ui/templates')
CORS(app)

# Register production-grade endpoints
from api.endpoints_v2 import api_v2
app.register_blueprint(api_v2, url_prefix='/api/v2')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'data/uploaded'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg'}

# Create directories
os.makedirs('data/uploaded', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Global variables
model = None
class_names = ['Legitimate', 'Suspicious']
uptime_start = datetime.now()

# Initialize version manager and validator
version_manager = ModelVersionManager()
validator = AudioValidator()

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
MAX_LENGTH = int(SAMPLE_RATE * DURATION)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_audio(file_path):
    """Load and preprocess audio file"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pad or trim
    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
    else:
        audio = audio[:MAX_LENGTH]

    return audio


def extract_mel_spectrogram(audio):
    """Extract mel-spectrogram features"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def preprocess_audio(file_path):
    """Complete preprocessing pipeline"""
    audio = load_audio(file_path)
    features = extract_mel_spectrogram(audio)
    # Add batch and channel dimensions
    features = features[np.newaxis, ..., np.newaxis]
    return features


def generate_visualizations(audio, sr):
    """Generate audio visualizations (Waveform, Mel Spectrogram, MFCC)"""
    visualizations = {}
    
    try:
        # 1. Waveform
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=sr)
        plt.title('Waveform')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['waveform'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Mel Spectrogram
        plt.figure(figsize=(10, 3))
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['spectrogram'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. MFCC
        plt.figure(figsize=(10, 3))
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['mfcc'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        
    return visualizations


def load_model_with_compat():
    """Load model with Keras compatibility mode"""
    global model

    try:
        model_paths = [
            'models/vishing_detector_savedmodel_v2',   # SavedModel (most stable)
            'models/vishing_detector_keras3.keras',    # Keras 3.x native format
            'models/vishing_detector_savedmodel',      # Legacy SavedModel
            'models/vishing_detector_compatible.h5',   # Converted H5
            'models/vishing_detector_final.h5',        # Original (Keras 2.x)
            'models/best_vishing_model.h5',            # Alternative (Keras 2.x)
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                logger.info(f"Attempting to load model from: {model_path}")

                try:
                    # Try with compile=False to avoid optimizer issues
                    model = keras.models.load_model(model_path, compile=False)

                    # Recompile with current Keras version
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    logger.info("SUCCESS: Model loaded successfully!")
                    logger.info(f"Model input shape: {model.input_shape}")
                    logger.info(f"Model output shape: {model.output_shape}")
                    return True

                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {str(e)}")
                    continue

        logger.error("Could not load model with any method")
        return False

    except Exception as e:
        logger.error(f"Error in load_model_with_compat: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Serve enhanced production dashboard"""
    return render_template('index_enhanced.html')


@app.route('/original')
def original_dashboard():
    """Serve original dashboard"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - uptime_start).total_seconds()

    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict class for single audio file"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: wav, mp3, flac, ogg'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"Processing file: {filename}")

        # Preprocess audio
        features = preprocess_audio(filepath)

        # Make prediction
        predictions = model.predict(features, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # Build result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'file_path': filename,
            'probabilities': {
                class_names[0]: float(predictions[0][0]),
                class_names[1]: float(predictions[0][1])
            }
        }

        # Generate visualizations
        try:
            audio = load_audio(filepath)
            viz = generate_visualizations(audio, SAMPLE_RATE)
            result['visualizations'] = viz
        except Exception as e:
            logger.error(f"Visualization error: {e}")


        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def predict_batch():
    """Predict classes for multiple audio files"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')

    if not files:
        return jsonify({'error': 'No files selected'}), 400

    try:
        results = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Preprocess and predict
                features = preprocess_audio(filepath)
                predictions = model.predict(features, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])

                results.append({
                    'file': filename,
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })

        logger.info(f"Batch prediction completed for {len(results)} files")

        return jsonify({
            'count': len(results),
            'predictions': results
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    info = {
        'model_type': 'CNN',
        'num_classes': 2,
        'class_names': class_names,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION,
        'n_mels': N_MELS
    }

    return jsonify(info), 200


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model metrics"""
    metrics_file = 'logs/metrics.json'

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics), 200
    else:
        return jsonify({'message': 'No metrics available'}), 404


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload new training data"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    class_label = request.form.get('class_label', 'unknown')

    try:
        # Create directory for this class
        class_dir = os.path.join('data/uploaded/training', class_label)
        os.makedirs(class_dir, exist_ok=True)

        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(class_dir, filename)
                file.save(filepath)
                saved_files.append(filename)

        logger.info(f"Uploaded {len(saved_files)} files for class '{class_label}'")

        return jsonify({
            'message': f'Uploaded {len(saved_files)} files',
            'class_label': class_label,
            'files': saved_files
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining in background"""
    retrainer = get_retrainer()

    # Check if already running
    status = retrainer.get_status()
    if status['status'] == 'running':
        return jsonify({
            'error': 'Retraining already in progress',
            'status': status
        }), 409

    # Get training parameters
    epochs = request.json.get('epochs', 30) if request.is_json else 30
    batch_size = request.json.get('batch_size', 32) if request.is_json else 32

    # Start retraining in background thread
    def retrain_background():
        logger.info(f"Starting background retraining (epochs={epochs}, batch_size={batch_size})")
        retrainer.retrain(epochs=epochs, batch_size=batch_size)

    thread = threading.Thread(target=retrain_background, daemon=True)
    thread.start()

    logger.info("Retraining triggered")

    return jsonify({
        'message': 'Retraining initiated in background',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'check_status_at': '/retrain_status'
    }), 202


@app.route('/retrain_status', methods=['GET'])
def get_retrain_status():
    """Get current retraining status"""
    retrainer = get_retrainer()
    status = retrainer.get_status()

    return jsonify(status), 200


@app.route('/debug')
def debug_info():
    """Debug endpoint to check system state"""
    files = []
    for root, dirs, filenames in os.walk('.'):
        for filename in filenames:
            if 'git' not in root:
                files.append(os.path.join(root, filename))
            
    return jsonify({
        'files': files[:100],  # Limit output
        'models_dir': os.listdir('models') if os.path.exists('models') else 'models dir not found',
        'cwd': os.getcwd(),
        'python_version': sys.version,
        'tensorflow_version': tf.__version__
    })

@app.route('/force_load')
def force_load():
    """Manually trigger model loading"""
    global model
    try:
        from src.model import AudioClassifier
        path = 'models/vishing_detector_keras3.keras'
        
        if not os.path.exists(path):
            return jsonify({'error': f'File not found: {path}'}), 404
            
        loaded_instance = AudioClassifier.load_model(path)
        model = loaded_instance.model
        
        return jsonify({
            'status': 'success', 
            'message': 'Model loaded successfully'
        }
)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ============================================================================
# MODULE-LEVEL INITIALIZATION (runs when Gunicorn imports this module)
# ============================================================================
logger.info("="*60)
logger.info("Initializing Voice Phishing Detection API...")
logger.info("="*60)

# Load model at module level so it works with Gunicorn
model_loaded = load_model_with_compat()

if model_loaded:
    logger.info("="*60)
    logger.info("SUCCESS: Model loaded - Ready to serve predictions!")
    logger.info("="*60)
else:
    logger.warning("="*60)
    logger.warning("WARNING: Model not loaded - API will return 503 errors")
    logger.warning("="*60)

# ============================================================================
# DEVELOPMENT SERVER (only runs when executed directly)
# ============================================================================
if __name__ == '__main__':
    logger.info("Starting development server...")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable auto-reloader to prevent interruptions during audio processing
    )
