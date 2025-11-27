<#
.SYNOPSIS
    Automates the deployment of the Voice Phishing Detection API to Google Cloud Run.

.DESCRIPTION
    This script helps set up the project, create the Artifact Registry, build the container,
    and deploy it to Cloud Run.

.NOTES
    Requires Google Cloud SDK (gcloud) to be installed and authenticated.
#>

param(
    [string]$ProjectId = $env:PROJECT_ID,
    [string]$Region = "us-central1",
    [string]$RepoName = "voice-phishing-repo",
    [string]$ImageName = "voice-phishing-api",
    [string]$ServiceName = "voice-phishing-detector",
    [string]$Tag = "",
    [switch]$AllowUnauthenticated,
    [switch]$CI,
    [string]$ServiceAccountKeyFile = "",
    [switch]$SkipVenv
)

# Use "Continue" so that stderr output (status messages) doesn't crash the script.
# We will check $LASTEXITCODE to handle actual errors.
$ErrorActionPreference = "Continue"

function Write-Color([string]$text, [ConsoleColor]$color) {
    Write-Host $text -ForegroundColor $color
}

function Check-LastExitCode {
    if ($LASTEXITCODE -ne 0) {
        Write-Color "Error: The last command failed with exit code $LASTEXITCODE." Red
        exit 1
    }
}

Write-Color "=== Voice Phishing Detection API - GCP Deployment ===" Green

# Optionally activate virtual environment to help gcloud find python in local dev scenarios
if (-not $SkipVenv) {
    $venvActivate = Join-Path $PSScriptRoot "..\venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        Write-Color "Activating virtual environment..." Gray
        . $venvActivate
        try { $env:CLOUDSDK_PYTHON = (Get-Command python).Source } catch {}
        Write-Color "Using python: $($env:CLOUDSDK_PYTHON)" Gray
    }
    else {
        Write-Color "Virtual environment not found at $venvActivate (continuing)." Yellow
    }
}

# Check for gcloud
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Color "Error: Google Cloud SDK (gcloud) is not installed or not in PATH." Red
    Write-Color "Please install it from: https://cloud.google.com/sdk/docs/install" Yellow
    exit 1
}

# If a service account key is provided (CI), activate it
if ($ServiceAccountKeyFile -and (Test-Path $ServiceAccountKeyFile)) {
    Write-Color "Activating service account from key file..." Gray
    gcloud auth activate-service-account --key-file $ServiceAccountKeyFile
    Check-LastExitCode
}

# Resolve project id
if (-not $ProjectId) {
    $currentProject = gcloud config get-value project 2>$null
    if (-not $currentProject -or $currentProject -eq "(unset)") {
        if ($CI) {
            Write-Color "Error: Project ID not set and running in CI mode. Set PROJECT_ID env or pass -ProjectId." Red
            exit 1
        }
        Write-Color "No active Google Cloud project found." Yellow
        $ProjectId = Read-Host "Enter your Google Cloud Project ID"
        gcloud config set project $ProjectId
        Check-LastExitCode
    }
    else {
        Write-Color "Using current project: $currentProject" Cyan
        $ProjectId = $currentProject
    }
}

# Compute image tag
if (-not $Tag -or $Tag -eq "") {
    try {
        $gitShort = (git rev-parse --short HEAD 2>$null)
    }
    catch { $gitShort = $null }
    if ($gitShort) { $Tag = $gitShort } else { $Tag = "latest" }
}

$ImageTag = "${Region}-docker.pkg.dev/${ProjectId}/${RepoName}/${ImageName}:${Tag}"

Write-Color "Project: $ProjectId" Cyan
Write-Color "Region: $Region" Cyan
Write-Color "Repository: $RepoName" Cyan
Write-Color "Image: $ImageTag" Cyan

Write-Color "`nRequired IAM Roles (ensure the invoking identity has these):" Yellow
Write-Color " - roles/run.admin" Yellow
Write-Color " - roles/cloudbuild.builds.builder (or cloudbuild.builds.editor)" Yellow
Write-Color " - roles/artifactregistry.admin (or equivalent)" Yellow

if (-not $CI) {
    $confirm = Read-Host "Proceed with enabling APIs and deploying? (y/N)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-Color "Aborting as requested." Yellow
        exit 0
    }
}

# Enable APIs
Write-Color "`n1. Enabling necessary APIs..." Cyan
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project $ProjectId
Check-LastExitCode

# Create Repository (if not exists)
Write-Color "`n2. Checking Artifact Registry..." Cyan
$repoExists = gcloud artifacts repositories list --location=$Region --project $ProjectId --filter="name:$RepoName" --format="value(name)" 2>$null
if (-not $repoExists) {
    Write-Color "Creating repository '$RepoName'..." Yellow
    gcloud artifacts repositories create $RepoName `
        --repository-format=docker `
        --location=$Region `
        --description="Docker repository for Voice Phishing Detector" `
        --project $ProjectId
    Check-LastExitCode
}
else {
    Write-Color "Repository '$RepoName' already exists." Green
}

# Build and Submit
Write-Color "`n3. Building and submitting container image (this may take a few minutes)..." Cyan
gcloud builds submit --tag $ImageTag --project $ProjectId .
Check-LastExitCode

# Deploy
Write-Color "`n4. Deploying to Cloud Run..." Cyan
$allowFlag = ''
if ($AllowUnauthenticated) { $allowFlag = '--allow-unauthenticated' }

gcloud run deploy $ServiceName `
    --image $ImageTag `
    --platform managed `
    --region $Region `
    $allowFlag `
    --memory 8Gi `
    --cpu 4 `
    --project $ProjectId
Check-LastExitCode

Write-Color "`n=== Deployment Complete! ===" Green
Write-Color "Your service should now be live at the URL provided above." Cyan
