"""
Data preparation module for the Healthcare RAG system.
This module handles downloading and preprocessing the MedQuAD dataset.
"""
import os
import pandas as pd
import requests
from tqdm import tqdm
import json
from google.cloud import storage

def download_medquad_dataset(output_path):
    """
    Download the MedQuAD dataset.
    NOTE: This is a placeholder. In a real implementation, you would need to:
    1. Find the actual download link for MedQuAD
    2. Handle authentication if required
    3. Download and extract the dataset
    
    For demonstration purposes, we'll simulate downloading and create a sample dataset.
    """
    print(f"Downloading MedQuAD dataset to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simulating a small sample of MedQuAD format
    # In reality, you would download and parse the actual dataset
    sample_data = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, presence of ketones in urine, fatigue, irritability, blurred vision, slow-healing sores, and frequent infections."
        },
        {
            "question": "How is hypertension diagnosed?",
            "answer": "Hypertension is diagnosed when a person's blood pressure is consistently elevated. A healthcare provider will take multiple blood pressure readings at separate appointments before diagnosing hypertension. Normal blood pressure is below 120/80 mm Hg. Hypertension is defined as blood pressure above 130/80 mm Hg."
        },
        {
            "question": "What treatments are available for asthma?",
            "answer": "Asthma treatments include: 1) Quick-relief medications (bronchodilators) that rapidly open swollen airways, 2) Long-term control medications like inhaled corticosteroids that reduce airway inflammation, 3) Biologics for severe asthma, and 4) Bronchial thermoplasty for severe cases. Treatment plans usually involve identifying and avoiding triggers, monitoring breathing, and following an action plan."
        }
    ]
    
    # Create more sample data
    for i in range(20):
        sample_data.append({
            "question": f"Sample medical question #{i+1}?",
            "answer": f"Sample detailed medical answer for question #{i+1} with relevant healthcare information and terminology."
        })
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample dataset with {len(sample_data)} Q&A pairs at {output_path}")
    
    return output_path

def preprocess_dataset(input_path, output_path):
    """
    Preprocess the MedQuAD dataset into a format suitable for fine-tuning.
    Converts JSON to CSV with 'question' and 'answer' columns.
    """
    print(f"Preprocessing dataset from {input_path} to {output_path}...")
    
    # Load the JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure we have the expected columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' columns")
    
    # Save as CSV
    df.to_csv(output_path, index=False)
    
    print(f"Saved preprocessed dataset with {len(df)} rows to {output_path}")
    
    return output_path

def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """
    Upload a file to Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path)

    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")

def main():
    """Main function to execute the data preparation pipeline."""
    # Local paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_data_path = os.path.join(data_dir, 'raw', 'medquad.json')
    processed_data_path = os.path.join(data_dir, 'processed', 'medquad_qa.csv')
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Download the dataset
    download_medquad_dataset(raw_data_path)
    
    # Preprocess the dataset
    preprocess_dataset(raw_data_path, processed_data_path)
    
    # Example of uploading to GCS (commented out as it requires GCP credentials)
    # bucket_name = "your-gcp-project-rag-bucket"
    # upload_to_gcs(processed_data_path, bucket_name, "data/medquad_qa.csv")
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
