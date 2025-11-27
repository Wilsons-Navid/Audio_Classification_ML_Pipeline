import requests
import os

BASE_URL = 'http://localhost:5000/api/v2/upload_with_label'

def upload_files(class_label, folder_path, num_files=3):
    files_to_upload = []
    
    # Get first N files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    selected_files = all_files[:num_files]
    
    files_payload = []
    opened_files = []
    
    try:
        for fname in selected_files:
            fpath = os.path.join(folder_path, fname)
            f = open(fpath, 'rb')
            opened_files.append(f)
            files_payload.append(('files', (fname, f, 'audio/wav')))
            
        data = {'class_label': class_label}
        
        print(f"Uploading {len(files_payload)} files to {class_label}...")
        response = requests.post(BASE_URL, files=files_payload, data=data)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    finally:
        for f in opened_files:
            f.close()

if __name__ == '__main__':
    base_dir = r'C:\Users\LENOVO\Downloads\Audio_Classification_ML_Pipeline\retraining_samples'
    
    upload_files('Suspicious', os.path.join(base_dir, 'Suspicious'))
    upload_files('Legitimate', os.path.join(base_dir, 'Legitimate'))
