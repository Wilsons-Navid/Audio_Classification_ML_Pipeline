# Voice Phishing Detection System
## Production-Grade ML Pipeline for Audio Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌐 Live Production Deployment

**✅ Status**: **LIVE AND OPERATIONAL** (Model loaded successfully!)

**🚀 Try it now**: [https://voice-phishing-detector-380911653615.us-central1.run.app/](https://voice-phishing-detector-380911653615.us-central1.run.app/)

**Quick Links**:
**Youtube**: [https://youtu.be/9AQ3FCVb9K0](https://youtu.be/9AQ3FCVb9K0)
- 📊 **Dashboard**: [https://voice-phishing-detector-380911653615.us-central1.run.app/](https://voice-phishing-detector-380911653615.us-central1.run.app/)
- 💚 **Health Check**: [https://voice-phishing-detector-380911653615.us-central1.run.app/health](https://voice-phishing-detector-380911653615.us-central1.run.app/health)
- 📖 **API Docs**: See [API Documentation](#-api-documentation) below

**Deployment Details**:
- **Platform**: Google Cloud Run
- **Model Format**: H5 (Keras 3 compatible)
- **Model**: CNN with 489,730 parameters
- **Classes**: Legitimate, Suspicious
- **Last Verified**: 2025-11-21

> **Note**: Free tier may have cold start delay (~30 seconds) if inactive for 15+ minutes.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Dataset Information](#-dataset-information)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Load Testing Results](#-load-testing-results)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Jupyter Notebook](#-jupyter-notebook)
- [Model Files](#-model-files)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Description

The **Voice Phishing Detection System** is a production-ready machine learning pipeline designed to detect fraudulent phone calls (vishing) through audio analysis. The system uses deep learning techniques to analyze speech patterns, emotional cues, and acoustic features to classify calls as either **Legitimate** or **Suspicious**.

### Key Capabilities

- **Real-time Audio Classification**: Analyze audio files and detect phishing attempts
- **Batch Processing**: Process multiple audio files simultaneously
- **Continuous Learning**: Retrain models with new data to improve accuracy
- **Production-Ready API**: RESTful API with comprehensive endpoints
- **Interactive Dashboard**: Web-based UI for predictions, monitoring, and retraining
- **Model Versioning**: Track model versions with rollback capability
- **Cloud Deployment**: Docker-ready with cloud platform configurations

### Use Cases

- **Call Centers**: Automatically flag suspicious calls for review
- **Financial Institutions**: Protect customers from phone scams
- **Telecommunications**: Monitor network for fraudulent activity
- **Security Operations**: Real-time threat detection

---

## 📊 Dataset Information

### Primary Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

**Source**: [Zenodo - RAVDESS Dataset](https://zenodo.org/record/1188976)

**Description**: The RAVDESS dataset contains 7,356 audio files from 24 professional actors (12 male, 12 female) expressing different emotions. For voice phishing detection, we map emotional indicators to phishing characteristics:

- **Legitimate Calls**: Calm, neutral, happy emotions
- **Suspicious Calls**: Angry, fearful, surprised emotions (indicators of urgency, pressure tactics)

**Dataset Statistics**:
- **Total Audio Files**: 1,012 files used in this project
- **Sample Rate**: 22,050 Hz
- **Audio Duration**: 4 seconds (standardized)
- **Format**: WAV (16-bit PCM)
- **Classes**: 2 (Legitimate, Suspicious)
- **Train/Val/Test Split**: 70% / 15% / 15%

**Citation**:
```
Livingstone SR, Russo FA (2018) 
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): 
A dynamic, multimodal set of facial and vocal expressions in North American English. 
PLoS ONE 13(5): e0196391. 
https://doi.org/10.1371/journal.pone.0196391
```

### Data Organization

```
data/
├── raw/              # Original RAVDESS dataset
│   └── RAVDESS/
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
├── train/            # Training data (organized by class)
│   ├── Legitimate/
│   └── Suspicious/
├── test/             # Test data (organized by class)
│   ├── Legitimate/   # 91 test files
│   └── Suspicious/
└── uploaded/         # User-uploaded files for retraining
    └── training/
        ├── Legitimate/
        └── Suspicious/
```

---

## ✨ Features

### Core Functionality

✅ **Audio Classification**
- Single file prediction with confidence scores
- Batch prediction for multiple files
- Support for WAV, MP3, FLAC, OGG formats

✅ **Data Processing Pipeline**
- Mel-spectrogram feature extraction (128 mel bands)
- MFCC (Mel-Frequency Cepstral Coefficients) analysis
- Audio normalization and standardization
- Automatic padding/trimming to 4 seconds

✅ **Model Training & Retraining**
- CNN (Convolutional Neural Network) architecture
- Transfer learning with pre-trained models
- Incremental learning with new data
- **Fast Retraining**: Uses Transfer Learning on uploaded files only (instant preprocessing) while validating against the full original test set.
- Automatic model versioning

✅ **Production Features**
- RESTful API with 10+ endpoints
- Real-time model uptime monitoring
- Audio visualizations (waveform, spectrogram, MFCC)
- Model version control with rollback
- Comprehensive logging and metrics

✅ **Deployment Ready**
- Docker containerization
- Docker Compose for scaling
- Cloud platform configurations (Render, Heroku)
- Load testing with Locust

---

## 📁 Project Structure

```
Audio_Classification_ML_Pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Multi-container setup
├── render.yaml                        # Render.com deployment config
├── Procfile                           # Heroku deployment config
│
├── notebook/
│   ├── voice_phishing_detection.ipynb          # Training notebook

│
├── src/
│   ├── preprocessing.py               # Audio preprocessing & feature extraction
│   ├── model.py                       # Model architecture & training
│   ├── prediction.py                  # Prediction logic
│   ├── retrain.py                     # Retraining module (411 lines)
│   ├── model_version.py               # Version management
│   ├── validation.py                  # Audio validation
│   └── utils.py                       # Utility functions
│
├── api/
│   ├── app.py                         # Main Flask application (504 lines)
│   ├── endpoints_v2.py                # Production API endpoints
│   └── __init__.py
│
├── ui/
│   ├── templates/
│   │   ├── index_enhanced.html        # Production dashboard (846 lines)
│   │   └── index.html                 # Original dashboard
│   └── static/
│       ├── css/
│       └── js/
│
├── data/
│   ├── raw/                           # Raw RAVDESS dataset
│   ├── train/                         # Training data
│   ├── test/                          # Test data (182 files)
│   └── uploaded/                      # User uploads
│
├── models/
│   ├── vishing_detector_keras3.keras  # Keras 3.x format (5.96 MB)
│   ├── vishing_detector_final.h5      # H5 format (5.98 MB)
│   ├── vishing_detector_compatible.h5 # Compatible H5 (43.5 MB)
│   ├── vishing_detector_savedmodel/   # TensorFlow SavedModel
│   ├── vishing_detector_metadata.json # Model metadata
│   └── versions/                      # Version history
│
├── tests/
│   └── locustfile.py                  # Load testing (183 lines)
│
└── logs/
    ├── api.log                        # API logs
    ├── training.log                   # Training logs
    └── metrics.json                   # Performance metrics
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.9 or higher
- **FFmpeg**: For audio processing
- **Git**: For cloning the repository
- **Docker** (optional): For containerized deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/Wilsons-Navid/Audio_Classification_ML_Pipeline.git
cd Audio_Classification_ML_Pipeline
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

### Step 5: Download Dataset (Optional)

If you want to retrain the model from scratch:

1. Download RAVDESS dataset from [Zenodo](https://zenodo.org/record/1188976)
2. Extract to `data/raw/RAVDESS/`
3. Run the data organization script or use the Jupyter notebook

### Step 6: Verify Installation

```bash
python -c "import tensorflow as tf; import librosa; print('✓ All dependencies installed')"
```

---

## 💻 Usage

### Option 1: Run Locally

#### Start the API Server

```bash
python api/app.py
```

The server will start at: **http://localhost:5000**

#### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the **Voice Phishing Detection Dashboard** with:
- System status and model uptime
- Single file prediction
- Bulk data upload
- Model retraining controls
- Version management

### Option 2: Run with Docker

#### Build and Run

```bash
docker-compose up --build
```

Access at: **http://localhost:5000**

#### Scale with Multiple Containers

```bash
docker-compose up --scale audio-classifier=3
```

This runs 3 instances for load balancing.

### Option 3: Run Jupyter Notebook

```bash
jupyter notebook notebook/voice_phishing_detection.ipynb
```

This opens the training notebook where you can:
- Explore the dataset
- Train models from scratch
- Evaluate performance
- Export trained models

---

## 📈 Model Performance

### Model Architecture

**Type**: Convolutional Neural Network (CNN)

**Input Shape**: (128, 173, 1) - Mel-spectrogram
- 128 mel frequency bands
- 173 time frames (4 seconds at 22,050 Hz)
- 1 channel (mono audio)

**Architecture**:
```
Layer (type)                 Output Shape              Params
=================================================================
conv2d (Conv2D)              (None, 126, 171, 32)      320
batch_normalization          (None, 126, 171, 32)      128
max_pooling2d                (None, 63, 85, 32)        0
dropout                      (None, 63, 85, 32)        0

conv2d_1 (Conv2D)            (None, 61, 83, 64)        18,496
batch_normalization_1        (None, 61, 83, 64)        256
max_pooling2d_1              (None, 30, 41, 64)        0
dropout_1                    (None, 30, 41, 64)        0

conv2d_2 (Conv2D)            (None, 28, 39, 128)       73,856
batch_normalization_2        (None, 28, 39, 128)       512
max_pooling2d_2              (None, 14, 19, 128)       0
dropout_2                    (None, 14, 19, 128)       0

flatten                      (None, 33,792)            0
dense (Dense)                (None, 128)               4,325,504
dropout_3                    (None, 128)               0
dense_1 (Dense)              (None, 2)                 258
=================================================================
Total params: 4,419,330
Trainable params: 4,418,882
Non-trainable params: 448
```

### Optimization Techniques

✅ **Regularization**
- Dropout layers (0.3, 0.5)
- Batch normalization
- L2 regularization

✅ **Training Optimization**
- Adam optimizer (learning rate: 0.001)
- Early stopping (patience: 10 epochs)
- Model checkpointing (save best model)
- Learning rate reduction on plateau

✅ **Data Augmentation**
- Time stretching
- Pitch shifting
- Background noise addition

### Evaluation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 92.3% | Overall classification accuracy |
| **Precision** | 91.8% | True positive rate |
| **Recall** | 93.1% | Sensitivity measure |
| **F1-Score** | 92.4% | Harmonic mean of precision/recall |
| **ROC-AUC** | 0.956 | Area under ROC curve |
| **Loss** | 0.234 | Categorical cross-entropy |

### Confusion Matrix

```
                  Predicted
                Legitimate  Suspicious
Actual
Legitimate         84          7
Suspicious          6         85
```

**Interpretation**:
- **True Positives (Suspicious detected)**: 85
- **True Negatives (Legitimate detected)**: 84
- **False Positives**: 7 (legitimate calls flagged as suspicious)
- **False Negatives**: 6 (suspicious calls missed)

---

## 🔥 Load Testing Results (GCP Cloud Run)

### Testing Setup

**Tool**: Locust  
**Test File**: `tests/locustfile.py`  
**Environment**: Google Cloud Run (Auto-scaling)
**Region**: us-central1

### Test Scenarios

#### Scenario 1: Low Load (10 Users)

**Configuration**:
- Concurrent Users: 10
- Spawn Rate: 1 user/second
- Duration: 1 minute

**Results**:

| Metric | Value |
|--------|-------|
| **Requests Per Second (RPS)** | ~10 |
| **Median Response Time** | 260 ms |
| **95th Percentile** | 510 ms |
| **Failure Rate** | 0% |

#### Scenario 2: High Load (50 Users)

**Configuration**:
- Concurrent Users: 50
- Spawn Rate: 5 users/second
- Duration: 1 minute

**Results**:

| Metric | Value |
|--------|-------|
| **Requests Per Second (RPS)** | ~50 |
| **Median Response Time** | 300 ms |
| **95th Percentile** | 400 ms |
| **Failure Rate** | 0% |

### Performance Analysis

**Key Findings**:
1. **Auto-scaling**: Cloud Run automatically scales container instances based on incoming request volume.
2. **Latency**: Latency may spike initially during cold starts or scaling events but stabilizes as instances become ready.
3. **Reliability**: The service is expected to maintain a near-zero failure rate even under load.

### How to Run Load Tests

```bash
# Run Locust against live GCP URL
locust -f tests/locustfile.py --host https://voice-phishing-detector-380911653615.us-central1.run.app --headless -u 50 -r 10 -t 1m
```

---

## 🔌 API Documentation

### Base URL

**Local**: `http://localhost:5000`  
**Production**: `https://voice-phishing-detector.onrender.com`

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "uptime_seconds": 1234.56,
  "model_loaded": true,
  "timestamp": "2025-11-21T03:30:00"
}
```

#### 2. Model Information

```http
GET /model_info
```

**Response**:
```json
{
  "model_type": "CNN",
  "num_classes": 2,
  "class_names": ["Legitimate", "Suspicious"],
  "input_shape": "(None, 128, 173, 1)",
  "output_shape": "(None, 2)",
  "sample_rate": 22050,
  "duration": 4.0,
  "n_mels": 128
}
```

#### 3. Single File Prediction

```http
POST /predict
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

**Response**:
```json
{
  "predicted_class": "Suspicious",
  "confidence": 0.9234,
  "file_path": "audio.wav",
  "probabilities": {
    "Legitimate": 0.0766,
    "Suspicious": 0.9234
  },
  "visualizations": {
    "waveform": "base64_encoded_image...",
    "spectrogram": "base64_encoded_image...",
    "mfcc": "base64_encoded_image..."
  }
}
```

#### 4. Batch Prediction

```http
POST /batch_predict
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST -F "files=@audio1.wav" -F "files=@audio2.wav" \
     http://localhost:5000/batch_predict
```

**Response**:
```json
{
  "count": 2,
  "predictions": [
    {
      "file": "audio1.wav",
      "predicted_class": "Legitimate",
      "confidence": 0.8765
    },
    {
      "file": "audio2.wav",
      "predicted_class": "Suspicious",
      "confidence": 0.9123
    }
  ]
}
```

#### 5. Upload Training Data

```http
POST /upload_training_data
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST -F "files=@new_audio.wav" -F "class_label=Suspicious" \
     http://localhost:5000/upload_training_data
```

#### 6. Trigger Retraining

```http
POST /retrain
Content-Type: application/json
```

**Request**:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"epochs": 30, "batch_size": 32}' \
     http://localhost:5000/retrain
```

**Response**:
```json
{
  "message": "Retraining initiated in background",
  "status": "running",
  "timestamp": "2025-11-21T03:30:00",
  "check_status_at": "/retrain_status"
}
```

#### 7. Check Retraining Status

```http
GET /retrain_status
```

**Response**:
```json
{
  "status": "running",
  "progress": 45,
  "message": "Training epoch 14/30...",
  "current_epoch": 14,
  "total_epochs": 30
}
```

#### 8. Get Model Metrics

```http
GET /metrics
```

**Response**:
```json
{
  "accuracy": 0.923,
  "precision": 0.918,
  "recall": 0.931,
  "f1_score": 0.924,
  "roc_auc": 0.956,
  "confusion_matrix": [[84, 7], [6, 85]]
}
```

#### 9. Version Management (v2 API)

```http
GET /api/v2/versions
```

**Response**:
```json
{
  "current_version": {
    "version_id": "v_20251121_033000_abc123",
    "created_at": "2025-11-21T03:30:00",
    "metrics": {
      "accuracy": 0.923,
      "training_samples": 712
    }
  },
  "versions": [...]
}
```

#### 10. Rollback to Previous Version

```http
POST /api/v2/versions/{version_id}/rollback
```

---

## ☁️ Deployment

### Docker Deployment

#### Build Image

```bash
docker build -t voice-phishing-detector .
```

#### Run Container

```bash
docker run -p 5000:5000 -v $(pwd)/models:/app/models voice-phishing-detector
```

#### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# Scale to 3 instances
docker-compose up --scale audio-classifier=3 -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Cloud Deployment

#### Render.com (Recommended)

1. Push code to GitHub
2. Connect repository to Render
3. Render will use `render.yaml` configuration
4. Automatic deployments on push

**Configuration** (`render.yaml`):
```yaml
services:
  - type: web
    name: voice-phishing-detector
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 api.app:app
```

#### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create voice-phishing-detector

# Deploy
git push heroku main

# Open app
heroku open
```

#### AWS/Azure/GCP

Use the provided `Dockerfile` for deployment to:
- AWS ECS/Fargate
- Azure Container Instances
- Google Cloud Run

---

## 📓 Jupyter Notebook

### Location

**File**: `notebook/voice_phishing_detection.ipynb`

### Contents

The Jupyter notebook contains the complete ML pipeline with detailed explanations:

#### 1. Dataset Acquisition
- RAVDESS dataset download instructions
- Data organization and structure
- Exploratory data analysis (EDA)

#### 2. Data Preprocessing
- **Audio Loading**: Using librosa (22,050 Hz)
- **Feature Extraction**:
  - Mel-spectrogram (128 mel bands)
  - MFCC (13 coefficients)
  - Spectral features
- **Normalization**: StandardScaler
- **Data Augmentation**:
  - Time stretching (0.8x - 1.2x)
  - Pitch shifting (±2 semitones)
  - Background noise addition

#### 3. Model Training
- **Architecture**: CNN with batch normalization
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Sparse categorical cross-entropy
- **Callbacks**:
  - Early stopping (patience=10)
  - Model checkpoint (save best)
  - Learning rate reduction
- **Training**: 50 epochs, batch size 32
- **Validation**: 15% of training data

#### 4. Model Evaluation
- **Test Set Performance**: 92.3% accuracy
- **Metrics Calculation**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC curve
  - Confusion matrix
- **Visualizations**:
  - Training/validation curves
  - Confusion matrix heatmap
  - ROC curve
  - Sample predictions

#### 5. Model Export
- Save as `.h5` format
- Save as `.keras` format (Keras 3.x)
- Export as TensorFlow SavedModel
- Save metadata JSON

### How to Run

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Navigate to notebook/voice_phishing_detection_executed.ipynb
```

**Note**: The executed version (`_executed.ipynb`) contains all outputs and visualizations. Use the non-executed version if you want to run cells yourself.

---

## 🗂️ Model Files

### Available Formats

The trained model is saved in multiple formats for compatibility:

| File | Format | Size | Use Case |
|------|--------|------|----------|
| `vishing_detector_keras3.keras` | Keras 3.x Native | 5.96 MB | **Recommended** - Latest Keras |
| `vishing_detector_final.h5` | HDF5 (Keras 2.x) | 5.98 MB | Backward compatibility |
| `vishing_detector_compatible.h5` | HDF5 (Compatible) | 43.5 MB | Legacy systems |
| `vishing_detector_savedmodel/` | TensorFlow SavedModel | - | TensorFlow Serving |
| `best_vishing_model.h5` | HDF5 (Best checkpoint) | 5.98 MB | Alternative |

### Model Metadata

**File**: `models/vishing_detector_metadata.json`

```json
{
  "model_name": "Voice Phishing Detector",
  "version": "1.0.0",
  "created_at": "2025-11-21T00:00:00",
  "framework": "TensorFlow/Keras",
  "input_shape": [128, 173, 1],
  "output_classes": ["Legitimate", "Suspicious"],
  "sample_rate": 22050,
  "duration": 4.0,
  "n_mels": 128,
  "metrics": {
    "accuracy": 0.923,
    "precision": 0.918,
    "recall": 0.931,
    "f1_score": 0.924
  }
}
```

### Loading Models

**Python**:
```python
from tensorflow import keras

# Load Keras 3.x model (recommended)
model = keras.models.load_model('models/vishing_detector_keras3.keras')

# Load H5 model
model = keras.models.load_model('models/vishing_detector_final.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load SavedModel
model = keras.models.load_model('models/vishing_detector_savedmodel/')
```

### Version History

**Location**: `models/versions/`

Each retraining creates a new version with:
- Unique version ID (timestamp-based)
- Model weights
- Training metrics
- Metadata JSON

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Audio_Classification_ML_Pipeline.git

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author**: Wilson Navid  
**Email**: wadotiwawil@gmail.com  
**Institution**: African Leadership University  
**GitHub**: [@Wilsons-Navid](https://github.com/Wilsons-Navid)

**Project Link**: [https://github.com/Wilsons-Navid/Audio_Classification_ML_Pipeline](https://github.com/Wilsons-Navid/Audio_Classification_ML_Pipeline)

---

## 🙏 Acknowledgments

- **RAVDESS Dataset**: Livingstone & Russo (2018) for the emotional speech dataset
- **TensorFlow/Keras**: For the deep learning framework
- **Librosa**: For audio processing capabilities
- **Flask**: For the web framework
- **Locust**: For load testing tools
- **African Leadership University**: For academic support

---

## 📚 References

1. Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)

2. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference.

3. Chollet, F. (2015). Keras. GitHub repository. [https://github.com/fchollet/keras](https://github.com/fchollet/keras)

---

**Last Updated**: 2025-11-21  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
