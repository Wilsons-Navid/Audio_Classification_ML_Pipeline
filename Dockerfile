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

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=api/app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "api/app.py"]
