#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if PROJECT_ID and REGION are set
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ]; then
  echo "Error: PROJECT_ID and REGION environment variables must be set."
  echo "You can set them using: export PROJECT_ID=\"your-project-id\" and export REGION=\"your-region\""
  exit 1
fi

# --- Configuration ---
SERVICE_NAME="healthcare-rag-app"
# You might want to parameterize this or derive it from the branch/tag
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# --- Build the Docker image using Cloud Build ---
echo "Building Docker image: ${IMAGE_TAG}"
gcloud builds submit --tag "${IMAGE_TAG}" .

echo "Image built successfully."

# --- Deploy to Cloud Run ---
echo "Deploying service ${SERVICE_NAME} to Cloud Run in region ${REGION}"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_TAG}" \
  --region "${REGION}" \
  --platform "managed" \
  --allow-unauthenticated \
  --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},BUCKET_NAME=${BUCKET_NAME:-${PROJECT_ID}-rag-bucket}" # Add other necessary env vars here
  # Add --service-account=your-service-account@... if needed for GCP permissions
  # Add --memory=... and --cpu=... if needed

echo "Deployment complete."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)')
echo "Service URL: ${SERVICE_URL}"
