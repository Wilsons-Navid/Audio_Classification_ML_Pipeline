# Fresh GCP deployment script
$project = "healthy-bonsai-443520-f7"
$service = "voice-phishing-detector"
$region = "us-central1"

# Set project
gcloud config set project $project

# Enable required services
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com

# Deploy the service from source
gcloud run deploy $service `
    --source . `
    --platform managed `
    --region $region `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --port 5000
