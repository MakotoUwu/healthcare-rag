#!/bin/bash
# Script to deploy the Healthcare RAG application to Google Cloud Run

set -e  # Exit on any error

# Load environment variables if .env file exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variables
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID environment variable is not set"
    echo "Please set it in .env file or export directly"
    exit 1
fi

if [ -z "$REGION" ]; then
    REGION="us-central1"
    echo "Using default region: $REGION"
fi

# Set service name
SERVICE_NAME="healthcare-rag-service"
IMAGE_NAME="gcr.io/$PROJECT_ID/healthcare-rag:latest"

echo "Deploying Healthcare RAG to Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Build the container image
echo "Building container image..."
gcloud builds submit --tag $IMAGE_NAME .

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars=PROJECT_ID=$PROJECT_ID,REGION=$REGION

echo "Deployment completed!"
echo "Your service should be available at the URL shown above."
