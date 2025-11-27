import os
import sys
import json
import logging

# Add src to path
sys.path.append(os.getcwd())

from src.model_version import ModelVersionManager

logging.basicConfig(level=logging.INFO)

def debug_registry():
    print("Initializing ModelVersionManager...")
    manager = ModelVersionManager()
    
    print(f"\nRegistry File: {manager.registry_file}")
    if os.path.exists(manager.registry_file):
        print("Registry file exists.")
        with open(manager.registry_file, 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print("Registry file does NOT exist.")
        
    versions = manager.list_versions()
    print(f"\nListed Versions: {len(versions)}")
    for v in versions:
        print(f"- {v['version_id']} ({v['created_at']})")

if __name__ == "__main__":
    debug_registry()
