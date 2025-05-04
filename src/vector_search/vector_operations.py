"""
Vector operations module for handling embedding generation and Vector Search operations.
"""
import os
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

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

def get_embedding_model(model_name=None):
    """
    Get the text embedding model from Vertex AI.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        TextEmbeddingModel instance
    """
    model_name = model_name or os.environ.get("EMBEDDING_MODEL", "textembedding-gecko@001")
    embed_model = TextEmbeddingModel.from_pretrained(model_name)
    print(f"Loaded embedding model: {model_name}")
    return embed_model

def generate_embeddings(texts, embed_model=None):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        embed_model: TextEmbeddingModel instance (if None, will be created)
        
    Returns:
        List of embeddings (each as a list of floats)
    """
    if embed_model is None:
        embed_model = get_embedding_model()
    
    # Process in batches to avoid API limits
    batch_size = 100
    all_embeddings = []
    
    print(f"Generating embeddings for {len(texts)} texts")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings_result = embed_model.get_embeddings(batch)
        batch_embeddings = [e.values for e in embeddings_result]
        all_embeddings.extend(batch_embeddings)
    
    print(f"Generated {len(all_embeddings)} embeddings, dimension: {len(all_embeddings[0]) if all_embeddings else 0}")
    return all_embeddings

def create_vector_index(
    dimensions=768,
    display_name="healthcare-doc-index",
    bucket_name=None,
    project_id=None,
    region=None,
):
    """
    Create a Vertex AI Vector Search index.
    
    Args:
        dimensions: Embedding dimensions
        display_name: Display name for the index
        bucket_name: GCS bucket name for index content
        project_id: GCP project ID
        region: GCP region
        
    Returns:
        The created index
    """
    project_id, region = init_vertex_ai(project_id, region)
    bucket_name = bucket_name or f"{project_id}-rag-bucket"
    
    print(f"Creating vector index {display_name} with {dimensions} dimensions")
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=f"gs://{bucket_name}/vector_index_content/",
        dimensions=dimensions,
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        shard_size="SHARD_SIZE_SMALL",
        index_update_method="STREAM_UPDATE"
    )
    
    index.wait()
    print(f"Created index: {index.resource_name}")
    return index

def create_index_endpoint(
    display_name="healthcare-index-endpoint",
    project_id=None,
    region=None,
):
    """
    Create a Vertex AI Vector Search index endpoint.
    
    Args:
        display_name: Display name for the endpoint
        project_id: GCP project ID
        region: GCP region
        
    Returns:
        The created index endpoint
    """
    init_vertex_ai(project_id, region)
    
    print(f"Creating index endpoint {display_name}")
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=display_name,
        public_endpoint_enabled=True
    )
    
    index_endpoint.wait()
    print(f"Created index endpoint: {index_endpoint.resource_name}")
    return index_endpoint

def deploy_index(
    index,
    index_endpoint,
    deployed_index_id="healthcare_index_1",
    machine_type="e2-standard-2",
    min_replica_count=1,
    max_replica_count=1,
):
    """
    Deploy an index to an endpoint.
    
    Args:
        index: The index to deploy
        index_endpoint: The endpoint to deploy to
        deployed_index_id: ID for the deployed index
        machine_type: Machine type for hosting
        min_replica_count: Minimum number of replicas
        max_replica_count: Maximum number of replicas
        
    Returns:
        Deployment status
    """
    print(f"Deploying index to endpoint with ID {deployed_index_id}")
    deployment = index_endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count
    )
    
    print(f"Deployed index with ID {deployed_index_id} to endpoint")
    return deployment

def upsert_vectors(
    index,
    doc_ids,
    doc_embeddings,
    chunk_size=100,
):
    """
    Upsert document vectors into the index.
    
    Args:
        index: The index to upsert into
        doc_ids: List of document IDs
        doc_embeddings: List of document embeddings
        chunk_size: Size of each batch
        
    Returns:
        Upsert status
    """
    if len(doc_ids) != len(doc_embeddings):
        raise ValueError("Number of document IDs must match number of embeddings")
    
    # Prepare datapoints for upsert
    datapoints = [
        aiplatform.MatchingEngineIndexDatapoint(
            datapoint_id=doc_ids[i],
            feature_vector=doc_embeddings[i]
        )
        for i in range(len(doc_embeddings))
    ]
    
    # Upsert in batches
    print(f"Upserting {len(datapoints)} vectors into index in batches of {chunk_size}")
    for j in range(0, len(datapoints), chunk_size):
        batch = datapoints[j:j+chunk_size]
        index.upsert_datapoints(datapoints=batch)
    
    print(f"Upserted {len(datapoints)} vectors into index")
    return True

def search_similar_documents(
    index_endpoint,
    query_vector,
    deployed_index_id="healthcare_index_1",
    num_neighbors=3,
):
    """
    Search for similar documents using a query vector.
    
    Args:
        index_endpoint: The index endpoint to query
        query_vector: The query embedding vector
        deployed_index_id: ID of the deployed index
        num_neighbors: Number of neighbors to return
        
    Returns:
        List of search results (IDs and distances)
    """
    print(f"Searching for {num_neighbors} similar documents")
    search_response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_vector],
        num_neighbors=num_neighbors
    )
    
    results = []
    if search_response and search_response[0]:
        for neighbor in search_response[0]:
            results.append({
                "id": neighbor.id,
                "distance": neighbor.distance
            })
    
    print(f"Found {len(results)} similar documents")
    return results

def main():
    """Example usage of vector operations."""
    print("This module provides vector operations for Vertex AI Vector Search.")
    print("Import and use specific functions as needed in your application.")

if __name__ == "__main__":
    main()
