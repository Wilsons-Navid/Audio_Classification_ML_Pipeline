"""
RAVDESS Data Organization Script

This script parses the RAVDESS dataset filenames and organizes them into 
'Legitimate' and 'Suspicious' classes based on emotional content.

Mapping:
- Legitimate: Neutral (01), Calm (02), Happy (03)
- Suspicious: Angry (05), Fearful (06), Surprised (08)
- Ignored: Sad (04), Disgust (07)

Output Structure:
data/
  train/
    Legitimate/
    Suspicious/
  test/
    Legitimate/
    Suspicious/
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob

def organize_ravdess(raw_path, output_path, test_size=0.15, random_state=42):
    print("Organizing RAVDESS dataset...")
    
    # Define emotion codes
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    LEGITIMATE_EMOTIONS = ['01', '02', '03']
    SUSPICIOUS_EMOTIONS = ['05', '06', '08']
    
    # Collectors
    files_by_class = {
        'Legitimate': [],
        'Suspicious': []
    }
    
    # Find all WAV files
    # RAVDESS structure: Actor_01/03-01-01-01-01-01-01.wav
    search_pattern = os.path.join(raw_path, "**", "*.wav")
    all_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(all_files)} total audio files.")
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        
        if len(parts) != 7:
            continue
            
        emotion_code = parts[2]
        
        if emotion_code in LEGITIMATE_EMOTIONS:
            files_by_class['Legitimate'].append(file_path)
        elif emotion_code in SUSPICIOUS_EMOTIONS:
            files_by_class['Suspicious'].append(file_path)
            
    print(f"Class distribution: Legitimate={len(files_by_class['Legitimate'])}, Suspicious={len(files_by_class['Suspicious'])}")
    
    # Create output directories
    for split in ['train', 'test']:
        for class_name in ['Legitimate', 'Suspicious']:
            os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)
            
    # Split and copy
    total_copied = 0
    for class_name, files in files_by_class.items():
        train_files, test_files = train_test_split(
            files, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Copy Train
        for f in train_files:
            shutil.copy2(f, os.path.join(output_path, 'train', class_name, os.path.basename(f)))
            total_copied += 1
            
        # Copy Test
        for f in test_files:
            shutil.copy2(f, os.path.join(output_path, 'test', class_name, os.path.basename(f)))
            total_copied += 1
            
    print(f"Successfully organized {total_copied} files into {output_path}/train and {output_path}/test")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw"
    OUTPUT_PATH = "data"
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} does not exist.")
    else:
        organize_ravdess(RAW_DATA_PATH, OUTPUT_PATH)
