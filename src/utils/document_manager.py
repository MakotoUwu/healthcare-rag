"""
Document manager for the Healthcare RAG system.
Handles loading, storing, and retrieving document content for the vector database.
"""
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from google.cloud import storage

class DocumentManager:
    """Manages documents for the Healthcare RAG system."""
    
    def __init__(self, local_content_path: Optional[str] = None):
        """
        Initialize the document manager.
        
        Args:
            local_content_path: Path to locally store document content mappings
        """
        self.doc_id_to_text: Dict[str, str] = {}
        self.local_content_path = local_content_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'processed', 'document_contents.json'
        )
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.local_content_path), exist_ok=True)
        
    def load_from_csv(self, csv_path: str, question_col: str = 'question', answer_col: str = 'answer') -> int:
        """
        Load document content from a CSV file with questions and answers.
        
        Args:
            csv_path: Path to the CSV file
            question_col: Name of the question column
            answer_col: Name of the answer column
            
        Returns:
            Number of documents loaded
        """
        df = pd.read_csv(csv_path)
        
        # Generate document IDs and map to content
        count = 0
        for idx, row in df.iterrows():
            doc_id = f"doc_{idx}"
            question = row[question_col]
            answer = row[answer_col]
            
            # Store the answer as document content with question as metadata
            self.doc_id_to_text[doc_id] = answer
            count += 1
            
        print(f"Loaded {count} documents from {csv_path}")
        return count
    
    def load_from_jsonl(self, jsonl_path: str, question_key: str = 'question', answer_key: str = 'answer') -> int:
        """
        Load document content from a JSONL file with questions and answers.
        
        Args:
            jsonl_path: Path to the JSONL file
            question_key: Key for the question field
            answer_key: Key for the answer field
            
        Returns:
            Number of documents loaded
        """
        count = 0
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                doc_id = f"doc_{count}"
                answer = item.get(answer_key, "")
                
                # Store the answer as document content
                self.doc_id_to_text[doc_id] = answer
                count += 1
                
        print(f"Loaded {count} documents from {jsonl_path}")
        return count
                
    def save_to_local(self) -> str:
        """
        Save the document content mapping to a local JSON file.
        
        Returns:
            Path to the saved file
        """
        with open(self.local_content_path, 'w') as f:
            json.dump(self.doc_id_to_text, f, indent=2)
            
        print(f"Saved {len(self.doc_id_to_text)} document mappings to {self.local_content_path}")
        return self.local_content_path
    
    def load_from_local(self) -> int:
        """
        Load document content mapping from a local JSON file.
        
        Returns:
            Number of documents loaded
        """
        if os.path.exists(self.local_content_path):
            with open(self.local_content_path, 'r') as f:
                self.doc_id_to_text = json.load(f)
                
            print(f"Loaded {len(self.doc_id_to_text)} document mappings from {self.local_content_path}")
            return len(self.doc_id_to_text)
        else:
            print(f"Local content file not found: {self.local_content_path}")
            return 0
    
    def upload_to_gcs(self, bucket_name: str, blob_name: str = 'data/document_contents.json') -> str:
        """
        Upload document content mapping to Google Cloud Storage.
        
        Args:
            bucket_name: GCS bucket name
            blob_name: Name of the blob in the bucket
            
        Returns:
            GCS URI of the uploaded file
        """
        # Save to local first
        self.save_to_local()
        
        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(self.local_content_path)
        
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        print(f"Uploaded document content mapping to {gcs_uri}")
        return gcs_uri
    
    def download_from_gcs(self, bucket_name: str, blob_name: str = 'data/document_contents.json') -> int:
        """
        Download document content mapping from Google Cloud Storage.
        
        Args:
            bucket_name: GCS bucket name
            blob_name: Name of the blob in the bucket
            
        Returns:
            Number of documents loaded
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.download_to_filename(self.local_content_path)
        
        # Load from the downloaded file
        return self.load_from_local()
    
    def get_document_text(self, doc_id: str) -> str:
        """
        Get the text content of a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text content or empty string if not found
        """
        return self.doc_id_to_text.get(doc_id, "")
    
    def get_all_documents(self) -> List[Tuple[str, str]]:
        """
        Get all document IDs and their text content.
        
        Returns:
            List of (doc_id, text) tuples
        """
        return [(doc_id, text) for doc_id, text in self.doc_id_to_text.items()]
    
    def add_document(self, doc_id: str, text: str) -> None:
        """
        Add a new document or update an existing one.
        
        Args:
            doc_id: Document ID
            text: Document text content
        """
        self.doc_id_to_text[doc_id] = text
        
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was removed, False if not found
        """
        if doc_id in self.doc_id_to_text:
            del self.doc_id_to_text[doc_id]
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all document mappings."""
        self.doc_id_to_text = {}
        print("Cleared all document mappings")


def main():
    """Example usage of the DocumentManager."""
    # Local file paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    csv_path = os.path.join(data_dir, 'processed', 'medquad_qa.csv')
    
    # Create a DocumentManager instance
    doc_manager = DocumentManager()
    
    # Check if the CSV file exists
    if os.path.exists(csv_path):
        # Load documents from CSV
        doc_manager.load_from_csv(csv_path)
        
        # Save to local file
        doc_manager.save_to_local()
        
        print(f"Document Manager loaded with {len(doc_manager.doc_id_to_text)} documents")
    else:
        print(f"CSV file not found: {csv_path}")
        print("Run data_prep.py first to generate the dataset")

if __name__ == "__main__":
    main()
