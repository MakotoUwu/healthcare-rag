"""
Vertex AI operations module for submitting training jobs and deploying models.
"""
import os
from google.cloud import aiplatform

def init_vertex_ai(project_id=None, region=None):
    """
    Initialize Vertex AI client with project and region.
    Uses environment variables if not provided.
    """
    project_id = project_id or os.environ.get("PROJECT_ID")
    region = region or os.environ.get("REGION", "us-central1")
    
    if not project_id:
        raise ValueError("PROJECT_ID must be provided or set as an environment variable")
    
    aiplatform.init(project=project_id, location=region)
    print(f"Initialized Vertex AI for project {project_id} in {region}")
    
    return project_id, region

def submit_training_job(
    script_path,
    data_path,
    output_dir,
    project_id=None,
    region=None,
    bucket_name=None,
    display_name="flan-t5-medical-finetune",
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
):
    """
    Submit a custom training job to Vertex AI.
    
    Args:
        script_path: Path to the training script
        data_path: Path to the training data in Cloud Storage
        output_dir: Path to save model artifacts in Cloud Storage
        project_id: GCP project ID
        region: GCP region
        bucket_name: GCS bucket name
        display_name: Display name for the training job
        machine_type: Machine type for training
        accelerator_type: Accelerator type for training
        accelerator_count: Number of accelerators
        
    Returns:
        The trained model as a Vertex AI Model resource
    """
    project_id, region = init_vertex_ai(project_id, region)
    bucket_name = bucket_name or f"{project_id}-rag-bucket"
    
    # Define training job with container
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=script_path,
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
        requirements=["transformers==4.33.0", "datasets==2.14.0", "pandas==2.0.3"],
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest"
    )
    
    # Launch the training job
    print(f"Submitting training job with {machine_type} and {accelerator_count} {accelerator_type}")
    model = job.run(
        args=[],
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        base_output_dir=output_dir,
        environment_variables={
            "DATA_PATH": data_path,
            "OUTPUT_DIR": "model_output",
        }
    )
    
    print(f"Training job completed. Model artifact URI: {model.uri}")
    return model

def deploy_model(
    model,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1,
    project_id=None,
    region=None,
):
    """
    Deploy a trained model to a Vertex AI Endpoint.
    
    Args:
        model: Vertex AI Model resource to deploy
        machine_type: Machine type for serving
        min_replica_count: Minimum number of replicas
        max_replica_count: Maximum number of replicas
        project_id: GCP project ID
        region: GCP region
        
    Returns:
        The deployed endpoint
    """
    init_vertex_ai(project_id, region)
    
    print(f"Deploying model to endpoint with {machine_type}")
    endpoint = model.deploy(
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_split={"0": 100}
    )
    
    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint

def main():
    """Main function to demonstrate training and deployment."""
    # This is a demonstration - in production you would use actual paths and parameters
    project_id = os.environ.get("PROJECT_ID", "your-project-id")
    bucket_name = f"{project_id}-rag-bucket"
    
    # Example paths
    script_path = "trainer.py"  # Local path to trainer.py
    data_path = f"gs://{bucket_name}/data/medquad_qa.csv"  # GCS path to data
    output_dir = f"gs://{bucket_name}/models/flan_t5_medical"  # GCS path for model output
    
    print("This script demonstrates submitting a training job to Vertex AI.")
    print("To actually run the job, you need to:")
    print("1. Set up the necessary GCP project and permissions")
    print("2. Update the paths and parameters")
    print("3. Remove these print statements and uncomment the code below")
    
    # Uncomment to actually run training and deployment:
    """
    model = submit_training_job(
        script_path=script_path,
        data_path=data_path,
        output_dir=output_dir,
        project_id=project_id,
        bucket_name=bucket_name
    )
    
    endpoint = deploy_model(model)
    """

if __name__ == "__main__":
    main()
