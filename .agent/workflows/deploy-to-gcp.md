---
description: Deploy the Voice Phishing Detection System to Google Cloud Platform
---

# Deploy to Google Cloud Platform

This workflow guides you through deploying the Voice Phishing Detection System to Google Cloud Platform using Cloud Run.

## Prerequisites

Before starting, ensure you have:

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **Docker** installed (for local testing)
4. **Project ID** ready (or create a new one)

## Step 1: Install and Configure gcloud CLI

If you haven't already installed the gcloud CLI:

**Windows (PowerShell):**
```powershell
# Download and install from: https://cloud.google.com/sdk/docs/install
# Or use Chocolatey:
choco install gcloudsdk
```

**Verify installation:**
```powershell
gcloud --version
```

## Step 2: Authenticate and Set Project

```powershell
# Login to Google Cloud
gcloud auth login

# List your projects
gcloud projects list

# Set your project (replace YOUR_PROJECT_ID)
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

## Step 3: Create .dockerignore File

To reduce build context size and avoid timeout errors, create a `.dockerignore` file:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Data directories (large files)
data/raw/
data/train/
data/test/
data/uploaded/

# Logs
logs/
*.log

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Documentation
*.md
!README.md

# Notebooks
notebook/
*.ipynb

# Tests
tests/

# Model versions (keep only main model)
models/versions/
models/*_savedmodel/
models/*.keras
```

## Step 4: Configure Environment Variables

Create a `.env.gcp` file for Cloud Run environment variables:

```
FLASK_ENV=production
PYTHONUNBUFFERED=1
MODEL_PATH=models/vishing_detector_final.h5
MAX_UPLOAD_SIZE=10485760
```

## Step 5: Build and Push Docker Image

```powershell
# Set variables
$PROJECT_ID = "YOUR_PROJECT_ID"
$REGION = "us-central1"  # or your preferred region
$SERVICE_NAME = "voice-phishing-detector"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Test locally (optional but recommended)
docker run -p 5000:5000 $IMAGE_NAME

# Push to Google Container Registry
docker push $IMAGE_NAME
```

**Alternative: Use Cloud Build (Recommended)**

```powershell
# Build directly on Google Cloud (avoids local Docker)
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
```

## Step 6: Deploy to Cloud Run

```powershell
# Deploy the service
gcloud run deploy $SERVICE_NAME `
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --max-instances 10 `
  --port 5000

# Get the service URL
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

## Step 7: Verify Deployment

```powershell
# Get the service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'

# Test health endpoint
curl "$SERVICE_URL/health"

# Test model info endpoint
curl "$SERVICE_URL/model_info"
```

## Step 8: Configure Custom Domain (Optional)

```powershell
# Map a custom domain
gcloud run domain-mappings create --service $SERVICE_NAME --domain your-domain.com --region $REGION
```

## Step 9: Set Up Continuous Deployment (Optional)

Connect your GitHub repository to Cloud Build for automatic deployments:

1. Go to Cloud Console → Cloud Build → Triggers
2. Click "Create Trigger"
3. Connect your GitHub repository
4. Configure trigger:
   - **Event**: Push to branch
   - **Branch**: `^main$`
   - **Build Configuration**: Cloud Build configuration file
5. Create `cloudbuild.yaml` in your repository root

## Troubleshooting

### Issue: Build Timeout

**Solution**: Ensure `.dockerignore` is properly configured to exclude large data files.

### Issue: Memory Errors

**Solution**: Increase memory allocation:
```powershell
gcloud run services update $SERVICE_NAME --memory 4Gi --region $REGION
```

### Issue: Cold Start Delays

**Solution**: Set minimum instances:
```powershell
gcloud run services update $SERVICE_NAME --min-instances 1 --region $REGION
```

### Issue: Model Not Loading

**Solution**: Check logs:
```powershell
gcloud run services logs read $SERVICE_NAME --region $REGION --limit 50
```

## Cost Optimization

- **Free Tier**: Cloud Run provides 2 million requests/month free
- **Minimum Instances**: Set to 0 for development to avoid charges
- **Memory**: Use 2Gi for production, 1Gi for testing
- **CPU**: Use 2 CPUs for better performance, 1 CPU for cost savings

## Monitoring and Logs

```powershell
# View logs
gcloud run services logs read $SERVICE_NAME --region $REGION

# Stream logs in real-time
gcloud run services logs tail $SERVICE_NAME --region $REGION

# View metrics in Cloud Console
# Navigate to: Cloud Run → Services → voice-phishing-detector → Metrics
```

## Cleanup

To delete the deployment:

```powershell
# Delete the Cloud Run service
gcloud run services delete $SERVICE_NAME --region $REGION

# Delete the container image
gcloud container images delete gcr.io/$PROJECT_ID/$SERVICE_NAME
```

## Next Steps

1. Set up monitoring and alerting
2. Configure CI/CD pipeline
3. Implement authentication if needed
4. Set up custom domain
5. Configure Cloud CDN for static assets
