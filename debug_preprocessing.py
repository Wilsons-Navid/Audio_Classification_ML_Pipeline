import os
import sys
import numpy as np
import librosa
import logging

# Add src to path
sys.path.append(os.getcwd())

from src.preprocessing import AudioPreprocessor

logging.basicConfig(level=logging.INFO)

def debug_single_file():
    file_path = r"C:\Users\LENOVO\Downloads\Audio_Classification_ML_Pipeline\retraining_samples\Suspicious\03-02-05-01-01-01-02.wav"
    
    print(f"Testing file: {file_path}")
    if not os.path.exists(file_path):
        print("File does not exist!")
        return

    preprocessor = AudioPreprocessor()
    
    print("\n--- Step 1: load_audio ---")
    try:
        audio = preprocessor.load_audio(file_path)
        print(f"Audio type: {type(audio)}")
        print(f"Audio shape: {audio.shape}")
        print(f"Audio dtype: {audio.dtype}")
    except Exception as e:
        print(f"load_audio failed: {e}")
        return

    print("\n--- Step 2: pad_or_trim ---")
    try:
        audio = preprocessor.pad_or_trim(audio)
        print(f"Padded audio shape: {audio.shape}")
    except Exception as e:
        print(f"pad_or_trim failed: {e}")
        return

    print("\n--- Step 3: extract_features ---")
    try:
        features = preprocessor.extract_features(audio, feature_type='mfcc')
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"extract_features failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_file()
