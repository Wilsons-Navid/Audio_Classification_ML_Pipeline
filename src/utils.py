"""
Utility Functions Module

This module provides utility functions for:
- Data loading
- Visualization
- Metrics calculation
- File operations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory(directory: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Directory created/verified: {directory}")


def save_json(data: Dict, file_path: str) -> None:
    """Save dictionary to JSON file"""
    create_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {file_path}")
    return data


def plot_training_history(history, save_path: str = None) -> None:
    """
    Plot training history

    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        create_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None
) -> None:
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        create_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")

    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: str = None
) -> None:
    """
    Plot class distribution

    Args:
        labels: Array of class labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique)), counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        create_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")

    plt.show()


def get_file_size(file_path: str) -> str:
    """Get human-readable file size"""
    size_bytes = os.path.getsize(file_path)

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} TB"


def count_audio_files(directory: str) -> Dict[str, int]:
    """
    Count audio files by extension in directory

    Args:
        directory: Path to directory

    Returns:
        Dictionary of extension counts
    """
    extensions = {'.wav': 0, '.mp3': 0, '.flac': 0, '.ogg': 0}

    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                extensions[ext] += 1

    total = sum(extensions.values())
    extensions['total'] = total

    return extensions


def log_metrics(metrics: Dict, log_file: str = 'logs/metrics.json') -> None:
    """
    Log metrics to file with timestamp

    Args:
        metrics: Dictionary of metrics
        log_file: Path to log file
    """
    create_directory(os.path.dirname(log_file))

    # Add timestamp
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }

    # Append to log file
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

    logger.info(f"Metrics logged to {log_file}")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    print("Utils module loaded successfully!")
