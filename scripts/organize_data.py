"""
Data Organization Script

This script organizes raw audio data into train/test splits
Supports different dataset formats
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict


def organize_urbansound8k(raw_data_path, output_path, test_size=0.2, random_state=42):
    """
    Organize UrbanSound8K dataset into train/test splits

    Args:
        raw_data_path: Path to raw UrbanSound8K data
        output_path: Output path for organized data
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
    """
    print("Organizing UrbanSound8K dataset...")

    # UrbanSound8K has folders: fold1, fold2, ..., fold10
    # Each fold contains audio files

    # Get metadata if available
    metadata_path = os.path.join(raw_data_path, 'metadata', 'UrbanSound8K.csv')

    if os.path.exists(metadata_path):
        # Use metadata to organize by class
        import pandas as pd
        metadata = pd.read_csv(metadata_path)

        # Group by class
        classes = metadata['class'].unique()

        train_files = []
        test_files = []

        for class_name in classes:
            class_files = metadata[metadata['class'] == class_name]

            # Split
            train_df, test_df = train_test_split(
                class_files,
                test_size=test_size,
                random_state=random_state
            )

            train_files.extend(train_df.to_dict('records'))
            test_files.extend(test_df.to_dict('records'))

        # Copy files to organized structure
        for split, files in [('train', train_files), ('test', test_files)]:
            for file_info in files:
                src_path = os.path.join(
                    raw_data_path,
                    f"fold{file_info['fold']}",
                    file_info['slice_file_name']
                )

                class_name = file_info['class']
                dest_dir = os.path.join(output_path, split, class_name)
                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, file_info['slice_file_name'])

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dest_path)

        print(f"Organized {len(train_files)} train files and {len(test_files)} test files")

    else:
        # Organize without metadata
        print("Metadata not found, organizing by fold structure...")
        organize_by_folders(raw_data_path, output_path, test_size, random_state)


def organize_esc50(raw_data_path, output_path, test_size=0.2, random_state=42):
    """
    Organize ESC-50 dataset into train/test splits

    Args:
        raw_data_path: Path to raw ESC-50 data
        output_path: Output path for organized data
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
    """
    print("Organizing ESC-50 dataset...")

    metadata_path = os.path.join(raw_data_path, 'meta', 'esc50.csv')

    if os.path.exists(metadata_path):
        import pandas as pd
        metadata = pd.read_csv(metadata_path)

        # Group by category
        categories = metadata['category'].unique()

        train_files = []
        test_files = []

        for category in categories:
            category_files = metadata[metadata['category'] == category]

            # Split
            train_df, test_df = train_test_split(
                category_files,
                test_size=test_size,
                random_state=random_state
            )

            train_files.extend(train_df.to_dict('records'))
            test_files.extend(test_df.to_dict('records'))

        # Copy files
        for split, files in [('train', train_files), ('test', test_files)]:
            for file_info in files:
                src_path = os.path.join(raw_data_path, 'audio', file_info['filename'])
                category = file_info['category']

                dest_dir = os.path.join(output_path, split, category)
                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, file_info['filename'])

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dest_path)

        print(f"Organized {len(train_files)} train files and {len(test_files)} test files")

    else:
        print("Metadata not found!")


def organize_speech_commands(raw_data_path, output_path, test_size=0.2, random_state=42):
    """
    Organize Speech Commands dataset into train/test splits

    Args:
        raw_data_path: Path to raw Speech Commands data
        output_path: Output path for organized data
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
    """
    print("Organizing Speech Commands dataset...")

    # Speech Commands has folders for each word
    # Each folder contains .wav files

    classes = [d for d in os.listdir(raw_data_path)
               if os.path.isdir(os.path.join(raw_data_path, d))
               and not d.startswith('_')]

    train_count = 0
    test_count = 0

    for class_name in classes:
        class_path = os.path.join(raw_data_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith('.wav')]

        # Split files
        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            random_state=random_state
        )

        # Copy train files
        train_dir = os.path.join(output_path, 'train', class_name)
        os.makedirs(train_dir, exist_ok=True)

        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_dir, file)
            shutil.copy2(src, dst)
            train_count += 1

        # Copy test files
        test_dir = os.path.join(output_path, 'test', class_name)
        os.makedirs(test_dir, exist_ok=True)

        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(test_dir, file)
            shutil.copy2(src, dst)
            test_count += 1

    print(f"Organized {train_count} train files and {test_count} test files")


def organize_by_folders(raw_data_path, output_path, test_size=0.2, random_state=42):
    """
    Generic organization for datasets organized in class folders

    Args:
        raw_data_path: Path to raw data
        output_path: Output path for organized data
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
    """
    print("Organizing dataset by folders...")

    # Find all audio files grouped by class (folder name)
    class_files = defaultdict(list)

    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                # Use parent directory as class name
                class_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                class_files[class_name].append(file_path)

    train_count = 0
    test_count = 0

    # Split each class
    for class_name, files in class_files.items():
        if not files:
            continue

        # Split
        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            random_state=random_state
        )

        # Copy train files
        train_dir = os.path.join(output_path, 'train', class_name)
        os.makedirs(train_dir, exist_ok=True)

        for file_path in train_files:
            dst = os.path.join(train_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dst)
            train_count += 1

        # Copy test files
        test_dir = os.path.join(output_path, 'test', class_name)
        os.makedirs(test_dir, exist_ok=True)

        for file_path in test_files:
            dst = os.path.join(test_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dst)
            test_count += 1

    print(f"Organized {train_count} train files and {test_count} test files")


def print_statistics(output_path):
    """Print statistics about organized data"""
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)

    for split in ['train', 'test']:
        split_path = os.path.join(output_path, split)

        if os.path.exists(split_path):
            classes = os.listdir(split_path)
            total_files = 0

            print(f"\n{split.upper()} Set:")
            for class_name in sorted(classes):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path)
                                if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))])
                    total_files += count
                    print(f"  {class_name}: {count} files")

            print(f"\nTotal {split} files: {total_files}")

    print("="*50)


if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = "data/raw"
    OUTPUT_PATH = "data"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Choose dataset type
    print("Available dataset types:")
    print("1. UrbanSound8K")
    print("2. ESC-50")
    print("3. Speech Commands")
    print("4. Generic (folder-based)")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '1':
        organize_urbansound8k(RAW_DATA_PATH, OUTPUT_PATH, TEST_SIZE, RANDOM_STATE)
    elif choice == '2':
        organize_esc50(RAW_DATA_PATH, OUTPUT_PATH, TEST_SIZE, RANDOM_STATE)
    elif choice == '3':
        organize_speech_commands(RAW_DATA_PATH, OUTPUT_PATH, TEST_SIZE, RANDOM_STATE)
    elif choice == '4':
        organize_by_folders(RAW_DATA_PATH, OUTPUT_PATH, TEST_SIZE, RANDOM_STATE)
    else:
        print("Invalid choice!")
        exit(1)

    # Print statistics
    print_statistics(OUTPUT_PATH)

    print("\n✓ Data organization complete!")
    print(f"✓ Training data: {OUTPUT_PATH}/train/")
    print(f"✓ Test data: {OUTPUT_PATH}/test/")
