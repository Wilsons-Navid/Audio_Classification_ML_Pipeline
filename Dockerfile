# Audio Classification ML Pipeline Dockerfile

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploaded data/train data/test models logs

# Expose port (Cloud Run uses PORT env var, defaults to 8080)
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=api/app.py
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp

# Run the application with proper worker configuration
# Use shell form to allow environment variable substitution
CMD gunicorn -w 2 -k gevent -b 0.0.0.0:${PORT:-8080} --timeout 120 api.app:app
