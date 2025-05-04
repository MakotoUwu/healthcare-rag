#!/usr/bin/env python
"""
Main orchestration module for the Healthcare RAG system.
This module ties together all components of the system.
"""
import os
import argparse
import logging
from dotenv import load_dotenv

# Import project modules
from src.data_prep import download_medquad_dataset, preprocess_dataset, upload_to_gcs
from src.utils.document_manager import DocumentManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_data_pipeline(args):
    """Set up the data processing pipeline."""
    logger.info("Setting up data pipeline...")
    
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_data_path = os.path.join(data_dir, 'raw', 'medquad.json')
    processed_data_path = os.path.join(data_dir, 'processed', 'medquad_qa.csv')
    
    # Create directories
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Download dataset
    if args.download_data or not os.path.exists(raw_data_path):
        download_medquad_dataset(raw_data_path)
    
    # Preprocess dataset
    if args.process_data or not os.path.exists(processed_data_path):
        preprocess_dataset(raw_data_path, processed_data_path)
    
    # Upload to GCS if credentials are available
    if args.upload_to_gcs and os.environ.get("PROJECT_ID"):
        bucket_name = os.environ.get("BUCKET_NAME", f"{os.environ['PROJECT_ID']}-rag-bucket")
        upload_to_gcs(processed_data_path, bucket_name, "data/medquad_qa.csv")
    
    # Load documents into document manager
    doc_manager = DocumentManager()
    doc_manager.load_from_csv(processed_data_path)
    doc_manager.save_to_local()
    
    logger.info("Data pipeline setup complete.")
    return processed_data_path

def setup_model_pipeline(args):
    """Set up the model training and deployment pipeline."""
    if not args.train_model:
        logger.info("Skipping model pipeline setup")
        return
    
    logger.info("Setting up model pipeline...")
    # This would typically import and run the Vertex AI training and deployment code
    # For now, we'll just log the step
    logger.info("To run model training and deployment:")
    logger.info("1. Set up GCP credentials")
    logger.info("2. Update .env with your PROJECT_ID and REGION")
    logger.info("3. Run: python -m src.model_training.vertex_ai_ops")
    
    logger.info("Model pipeline setup complete.")

def setup_vector_search(args):
    """Set up the vector search pipeline."""
    if not args.setup_vector_search:
        logger.info("Skipping vector search setup")
        return
    
    logger.info("Setting up vector search...")
    # This would typically import and run the vector search setup code
    # For now, we'll just log the step
    logger.info("To set up vector search:")
    logger.info("1. Set up GCP credentials")
    logger.info("2. Update .env with your PROJECT_ID and REGION")
    logger.info("3. Run: python -m src.vector_search.vector_operations")
    
    logger.info("Vector search setup complete.")

def run_web_app(args):
    """Run the web application."""
    if not args.run_app:
        logger.info("Skipping web app startup")
        return
    
    logger.info("Starting web application...")
    try:
        # Import here to avoid circular imports
        from src.app.app import app
        
        # Get port from environment or use default
        port = int(os.environ.get("PORT", 8080))
        
        # Run app
        app.run(host='0.0.0.0', port=port, debug=True)
    except ImportError:
        logger.error("Could not import Flask app. Make sure all dependencies are installed.")
        logger.info("To run the app manually: cd src/app && flask run --port=8080")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Healthcare RAG System Orchestration")
    
    # Data pipeline arguments
    parser.add_argument('--download-data', action='store_true', help='Download the dataset')
    parser.add_argument('--process-data', action='store_true', help='Process the dataset')
    parser.add_argument('--upload-to-gcs', action='store_true', help='Upload data to GCS')
    
    # Model pipeline arguments
    parser.add_argument('--train-model', action='store_true', help='Train the model')
    
    # Vector search arguments
    parser.add_argument('--setup-vector-search', action='store_true', help='Set up vector search')
    
    # Web app arguments
    parser.add_argument('--run-app', action='store_true', help='Run the web application')
    
    # All-in-one argument
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all steps
    if args.all:
        args.download_data = True
        args.process_data = True
        args.upload_to_gcs = True
        args.train_model = True
        args.setup_vector_search = True
        args.run_app = True
    
    return args

def main():
    """Main function to orchestrate the Healthcare RAG system."""
    logger.info("Healthcare RAG project started.")
    
    # Parse command line arguments
    args = parse_args()
    
    # Run each pipeline component
    data_path = setup_data_pipeline(args)
    setup_model_pipeline(args)
    setup_vector_search(args)
    run_web_app(args)
    
    logger.info("Healthcare RAG project completed.")

if __name__ == "__main__":
    main()
