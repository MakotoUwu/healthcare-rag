# app.py: Flask app for query handling
import os
import json
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

app = Flask(__name__)

# Initialize Vertex AI clients globally (reuse across requests for efficiency)
# Ensure these environment variables are set in your Cloud Run service or local environment
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
INDEX_ENDPOINT_NAME = os.environ.get("INDEX_ENDPOINT")  # Full resource name e.g. projects/.../indexEndpoints/...
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID") # e.g., "healthcare_index_1"
MODEL_ENDPOINT_NAME = os.environ.get("MODEL_ENDPOINT")  # Full resource name e.g. projects/.../endpoints/...
DOC_TEXTS_PATH = os.environ.get("DOC_TEXTS_PATH", "medquad_doc_texts.json") # Path to the document texts mapping

if not all([PROJECT_ID, REGION, INDEX_ENDPOINT_NAME, DEPLOYED_INDEX_ID, MODEL_ENDPOINT_NAME]):
    print("Warning: One or more environment variables (PROJECT_ID, REGION, INDEX_ENDPOINT, DEPLOYED_INDEX_ID, MODEL_ENDPOINT) are not set.")
    # Optionally, raise an error or use default values for local testing
    # raise ValueError("Required environment variables are missing.")

aiplatform.init(project=PROJECT_ID, location=REGION)
embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# Load or define the mapping from doc_ids to text (the knowledge base)
doc_texts = {}
try:
    # In a real application, load this from GCS or a database
    # For this example, assume a local JSON file
    with open(DOC_TEXTS_PATH, "r") as f:
        doc_texts = json.load(f)
    print(f"Loaded {len(doc_texts)} documents from {DOC_TEXTS_PATH}")
except FileNotFoundError:
    print(f"Warning: Document texts file not found at {DOC_TEXTS_PATH}. Context retrieval will fail.")
except Exception as e:
    print(f"Error loading document texts: {e}")

# Create endpoint objects if environment variables are set
index_endpoint = None
model_endpoint = None
if INDEX_ENDPOINT_NAME and DEPLOYED_INDEX_ID:
    try:
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_NAME)
        print(f"Initialized Index Endpoint: {INDEX_ENDPOINT_NAME}")
    except Exception as e:
        print(f"Error initializing Index Endpoint: {e}")
        index_endpoint = None # Ensure it's None if initialization fails

if MODEL_ENDPOINT_NAME:
    try:
        model_endpoint = aiplatform.Endpoint(endpoint_name=MODEL_ENDPOINT_NAME)
        print(f"Initialized Model Endpoint: {MODEL_ENDPOINT_NAME}")
    except Exception as e:
        print(f"Error initializing Model Endpoint: {e}")
        model_endpoint = None # Ensure it's None if initialization fails

@app.route("/query", methods=["POST"])
def answer_query():
    if not index_endpoint or not model_endpoint:
        return jsonify({"error": "Backend services (Index or Model Endpoint) not initialized. Check logs and environment variables."}), 503

    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Invalid request: 'question' field missing"}), 400

    query = data.get("question") or ""
    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        # 1. Embed the user query into vector
        embeddings_result = embed_model.get_embeddings([query])
        if not embeddings_result or not embeddings_result.embeddings:
             return jsonify({"error": "Failed to generate query embedding."}), 500
        query_embedding = embeddings_result.embeddings[0].values # Access the list of floats

        # 2. Search the vector index for similar content
        neighbors_response = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=3
        )

        if not neighbors_response or not neighbors_response[0]:
            retrieved_texts = []
            top_ids = []
            print("No neighbors found for the query.")
        else:
            top_neighbors = neighbors_response[0]  # list of MatchNeighbor objects for this query
            # Get IDs of top matches
            top_ids = [n.id for n in top_neighbors]
            retrieved_texts = [doc_texts.get(doc_id, "") for doc_id in top_ids]
            # Filter out empty strings if a doc_id wasn't found
            retrieved_texts = [text for text in retrieved_texts if text]

        # 3. Construct the prompt with retrieved context
        # Use top 2 available documents for context, handle cases with < 2 results
        context_block = "\n\n".join(retrieved_texts[:2])
        if not context_block:
            context_block = "No relevant context found."

        prompt = f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"

        # 4. Query the fine-tuned model endpoint for an answer
        prediction_response = model_endpoint.predict(instances=[{"content": prompt}])

        # The Vertex Endpoint returns a Prediction object; structure depends on model server
        # Assuming standard Vertex AI text model prediction format
        answer_text = ""
        if prediction_response.predictions and isinstance(prediction_response.predictions, list) and prediction_response.predictions[0]:
            # Adapt based on actual model output structure
            # Example: prediction_response.predictions[0]['content'] or prediction_response.predictions[0]
            prediction_content = prediction_response.predictions[0]
            if isinstance(prediction_content, dict) and 'content' in prediction_content:
                answer_text = prediction_content['content']
            elif isinstance(prediction_content, str):
                 answer_text = prediction_content
            else:
                print(f"Unexpected prediction format: {prediction_content}")
                answer_text = "Could not parse answer from model response."
        else:
            answer_text = "Model did not return a prediction."

        # 5. Return the answer
        return jsonify({"question": query, "answer": answer_text, "context_docs": top_ids})

    except Exception as e:
        print(f"Error processing query: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred processing the request."}), 500

if __name__ == "__main__":
    # Run Flask app locally (for testing)
    # Use environment variables for configuration
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)