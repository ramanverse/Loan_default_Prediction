#!/bin/bash

# Deployment script for Loan Default Prediction App

echo "🚀 Starting deployment process..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t loan-default-app .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Run container
echo "🏃 Starting container..."
docker run -d \
  -p 8501:8501 \
  -p 8000:8000 \
  --name loan-default-app \
  -v $(pwd)/models:/app/models \
  loan-default-app

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo "🌐 Streamlit app: http://localhost:8501"
    echo "🌐 API: http://localhost:8000"
    echo ""
    echo "To stop: docker stop loan-default-app"
    echo "To view logs: docker logs -f loan-default-app"
else
    echo "❌ Failed to start container!"
    exit 1
fi
