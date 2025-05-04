"""
Flask app for the Healthcare RAG Q&A system.
This app serves as the backend for the system, handling user queries
and returning answers based on the fine-tuned model and vector search.
"""
import os
import json
from flask import Flask, request, jsonify, render_template
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

app = Flask(__name__)

# Initialize Vertex AI clients globally (reuse across requests for efficiency)
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
FINETUNED_ENDPOINT_ID = os.environ.get("FINETUNED_ENDPOINT_ID")
VECTOR_INDEX_ENDPOINT_ID = os.environ.get("VECTOR_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "healthcare_index_1")

# Initialize Vertex AI
def init_app():
    """Initialize the Flask app with Vertex AI clients"""
    global embed_model, finetuned_endpoint, vector_endpoint
    
    # Check if we have all required environment variables
    if not PROJECT_ID:
        app.logger.warning("PROJECT_ID environment variable not set")
    if not FINETUNED_ENDPOINT_ID:
        app.logger.warning("FINETUNED_ENDPOINT_ID environment variable not set")
    if not VECTOR_INDEX_ENDPOINT_ID:
        app.logger.warning("VECTOR_INDEX_ENDPOINT_ID environment variable not set")
    
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        # Initialize embedding model
        embed_model = TextEmbeddingModel.from_pretrained(
            os.environ.get("EMBEDDING_MODEL", "textembedding-gecko@001")
        )
        
        # Initialize model endpoint if available
        if FINETUNED_ENDPOINT_ID:
            finetuned_endpoint = aiplatform.Endpoint(FINETUNED_ENDPOINT_ID)
        else:
            finetuned_endpoint = None
            app.logger.warning("Finetuned model endpoint not available")
        
        # Initialize vector endpoint if available
        if VECTOR_INDEX_ENDPOINT_ID:
            vector_endpoint = aiplatform.MatchingEngineIndexEndpoint(VECTOR_INDEX_ENDPOINT_ID)
        else:
            vector_endpoint = None
            app.logger.warning("Vector index endpoint not available")
            
        app.logger.info("Vertex AI clients initialized successfully")
        
    except Exception as e:
        app.logger.error(f"Error initializing Vertex AI clients: {e}")
        
        # For development purposes, set placeholder values
        # In production, you would want to fail fast
        embed_model = None
        finetuned_endpoint = None
        vector_endpoint = None

# Placeholder for mapping doc_id to text (load from database or storage)
doc_id_to_text = {}

# Load document contents from a JSON file (in production, use a database)
def load_document_contents():
    """Load document contents from a file or database"""
    global doc_id_to_text
    
    try:
        content_path = os.environ.get(
            "DOCUMENT_CONTENT_PATH", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                         "data", "processed", "document_contents.json")
        )
        
        if os.path.exists(content_path):
            with open(content_path, 'r') as f:
                doc_id_to_text = json.load(f)
            app.logger.info(f"Loaded {len(doc_id_to_text)} document contents from {content_path}")
        else:
            app.logger.warning(f"Document content file not found: {content_path}")
            
    except Exception as e:
        app.logger.error(f"Error loading document contents: {e}")

@app.route('/')
def index():
    """Render the homepage with the query interface"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "healthcare-rag-api"})

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle a query request"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    user_question = data.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Check if our services are available
    if not embed_model:
        return jsonify({"error": "Embedding model not available"}), 503
    if not vector_endpoint:
        return jsonify({"error": "Vector index not available"}), 503
    if not finetuned_endpoint:
        return jsonify({"error": "Finetuned model not available"}), 503

    try:
        # 1. Embed the query
        query_embedding_result = embed_model.get_embeddings([user_question])
        query_vector = query_embedding_result[0].values

        # 2. Vector Search
        NUM_NEIGHBORS = 3  # How many context chunks to retrieve
        search_response = vector_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vector],
            num_neighbors=NUM_NEIGHBORS
        )

        # 3. Fetch Content
        context_chunks = []
        if search_response and search_response[0]:  # Check if neighbors found
            for neighbor in search_response[0]:
                doc_id = neighbor.id
                # Fetch text using the ID
                chunk_text = doc_id_to_text.get(doc_id, f"Content for {doc_id} not found.")
                context_chunks.append(chunk_text)

        context = "\n".join(context_chunks)

        # 4. Construct Prompt
        prompt = f"Context: {context}\n\nQuestion: {user_question}\nAnswer:"

        # 5. Generate Answer
        prediction_response = finetuned_endpoint.predict(
            instances=[{"prompt": prompt}]
        )

        # Extract the generated answer (adjust based on actual response structure)
        generated_answer = prediction_response.predictions[0].get(
            "content", "Could not generate answer."
        )

        # 6. Return Answer
        return jsonify({
            "answer": generated_answer,
            "context": context_chunks,
            "queryTime": request.date,
        })

    except Exception as e:
        app.logger.error(f"Error processing query: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Initialize the app
init_app()
load_document_contents()

if __name__ == '__main__':
    # Run locally for testing (Cloud Run uses Gunicorn or similar)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
