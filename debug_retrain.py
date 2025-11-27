import os
import sys
import numpy as np
import logging

# Add src to path
sys.path.append(os.getcwd())

from src.retrain import ModelRetrainer

logging.basicConfig(level=logging.INFO)

def debug_data_loading():
    print("Initializing ModelRetrainer...")
    retrainer = ModelRetrainer()
    
    print("\n--- Debugging load_training_data ---")
    try:
        X_train, y_train, class_names = retrainer.load_training_data()
        
        print(f"\nClass names: {class_names}")
        print(f"X_train type: {type(X_train)}")
        print(f"X_train shape: {X_train.shape}")
        if len(X_train) > 0:
            print(f"First sample shape: {X_train[0].shape}")
            print(f"First sample min/max: {np.min(X_train[0])}, {np.max(X_train[0])}")
            
        print(f"y_train shape: {y_train.shape}")
        
        # Check for consistency
        shapes = [x.shape for x in X_train]
        unique_shapes = set(shapes)
        print(f"Unique sample shapes in X_train: {unique_shapes}")
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_loading()
