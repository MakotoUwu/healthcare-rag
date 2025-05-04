# Building a Domain-Specific RAG Q&A System with Vertex AI (Healthcare Example)

This guide walks through implementing a Retrieval-Augmented Generation (RAG) pipeline on Google Cloud Vertex AI. We will target the healthcare domain using an open medical Q&A dataset (MedQuAD), fine-tune a language model on this data, and integrate it with a vector database for retrieval. The steps include domain data selection, cloud infrastructure setup, model fine-tuning, vector index creation, a serving backend on Cloud Run, and deployment/testing.

## 1. Domain Selection & Dataset

**Domain:** Healthcare – a rich domain with high-quality open data. We use the MedQuAD dataset, which contains ~47k question-answer pairs curated from NIH websites. This provides authoritative Q&A examples on diseases, treatments, etc. Using such a dataset ensures our model learns domain-specific terminology and answer style.

**Dataset Source:** MedQuAD (Medical Question Answering Dataset) is open-source and includes 47,457 Q&A pairs from 12 NIH websites. Each entry has a medical question and an expert answer, covering topics like symptoms, treatments, and diagnostics.

**Download & Format:** Suppose the dataset is available as JSON or CSV (`question`, `answer` fields). We will preprocess it into a simple text format for fine-tuning (e.g., one JSONL or CSV with `input` = question, `output` = answer). We will also prepare the answer texts for building our knowledge vector index.

## 2. Google Cloud Infrastructure Setup

First, ensure you have a Google Cloud project and the Google Cloud SDK installed. Then enable required services and create cloud resources for storage and messaging:

```bash
# Set your project ID and region
PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
REGION="us-central1"   # or choose closest region with Vertex AI support
gcloud config set project $PROJECT_ID

# Enable necessary Google Cloud APIs for Vertex AI, Cloud Storage, Pub/Sub, and Cloud Run
gcloud services enable vertexai.googleapis.com storage.googleapis.com pubsub.googleapis.com run.googleapis.com

# Create a Cloud Storage bucket for storing dataset and model artifacts
BUCKET_NAME="${PROJECT_ID}-rag-bucket"
gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME

# (Optional) Create a Pub/Sub topic to trigger ingestion on new files (for document uploads)
PUBSUB_TOPIC="document-uploads"
gcloud pubsub topics create $PUBSUB_TOPIC

# (Optional) Configure the bucket to send notifications to Pub/Sub when new files are added
gsutil notification create -t $PUBSUB_TOPIC -f json gs://$BUCKET_NAME
```

**Explanation:** We enabled Vertex AI (for model training, endpoints, vector search), Cloud Storage (for data and models), Pub/Sub (for event-driven pipelines), and Cloud Run (for deploying our backend). We also created a storage bucket and set up a notification so that any new documents uploaded (if extending the system with PDF ingestion) will send a Pub/Sub message. This lays the groundwork for an ingestion pipeline to process and index new documents.

## 3. Fine-Tuning the Q&A Model on Domain Data

We will fine-tune a foundation language model on our healthcare Q&A data to specialize it. Vertex AI provides Training Pipelines to run custom training jobs on cloud hardware.

**Model Choice:** We'll fine-tune Flan-T5 (Google’s instruction-tuned T5 model) because it’s open-source and performs well on QA tasks. Specifically, we use the `google/flan-t5-base` checkpoint (250M parameters) for a balance of performance and cost. (Alternatively, one could use Google’s Gemma open models – Gemma models are lightweight and fine-tunable on Vertex AI – but here we demonstrate with Flan-T5 for concreteness.)

**Data Prep:** Ensure the dataset is cleaned and formatted. For example, create a CSV with columns `question` and `answer`, or a JSONL where each line is `{"question": "...", "answer": "..."}`. We might also prepend a prompt token (like `"question: ... answer:"`) during training if needed for formatting. Save this file to Cloud Storage (e.g., `gs://$BUCKET_NAME/data/medquad_qa.csv`).

**Custom Training Script:** We'll use the Hugging Face Transformers library within a Vertex AI custom training job. Below is a simplified training script (e.g., `trainer.py`) that we will run on Vertex AI. It loads the dataset, fine-tunes Flan-T5, and saves the model:

```python
# trainer.py: Fine-tune Flan-T5 on our Q&A data
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

# Hyperparameters and paths
MODEL_NAME = "google/flan-t5-base"
DATA_PATH = os.environ.get("DATA_PATH", "medquad_qa.csv")  # CSV with 'question','answer'
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "model_output")
EPOCHS = 3
BATCH_SIZE = 8

# Load dataset
df = pd.read_csv(DATA_PATH)
# Prepare training data in seq2seq format
train_encodings = []
labels = []
for q, a in zip(df["question"], df["answer"]):
    prompt = f"Question: {q}\nAnswer:"  # prompt format for model
    train_encodings.append(prompt)
    labels.append(a)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Tokenize the dataset
inputs = tokenizer(train_encodings, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
outputs = tokenizer(labels, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
# Replace pad token id in labels to -100 so they are ignored in loss
outputs["input_ids"][outputs["input_ids"] == tokenizer.pad_token_id] = -100

# Set up Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    logging_dir=f"{OUTPUT_DIR}/logs",
)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=list(zip(inputs["input_ids"], outputs["input_ids"]))) # Note: This dataset format is simplified for illustration
trainer.train()
trainer.save_model(OUTPUT_DIR)

```
*(Note: The original text mentioned `%%bash` and `python - <<'PYCODE'` which suggests running the python script via bash. I've formatted it as a standard Python code block. The original text also had a comment about using a proper Dataset object which I've kept implicitly by adding a note to the `train_dataset` line.)*

The above script is illustrative: In practice, you should use Hugging Face `datasets` for efficient data loading and the `Trainer` with a proper Dataset object. But it conveys the idea: we load Flan-T5, format each Q&A pair as a text-to-text example, and fine-tune the model to generate the answer given the question.

**Submitting the Training Job:** We use the Vertex AI SDK to launch this training on a GPU machine. The model and tokenizer will be downloaded and then fine-tuned on our data. After training, we upload the model artifacts to Vertex Model Registry and deploy to an endpoint.

```python
# Python code: Submit Vertex AI custom training job
from google.cloud import aiplatform
import os # Added import os

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
BUCKET_NAME = os.environ.get("BUCKET_NAME", f"{PROJECT_ID}-rag-bucket") # Added BUCKET_NAME definition based on earlier context

aiplatform.init(project=PROJECT_ID, location=REGION)

# Define training job with container (Vertex provides pre-built containers for training)
job = aiplatform.CustomTrainingJob(
    display_name="flan-t5-medical-finetune",
    script_path="trainer.py",           # path to the training script (uploaded or in context)
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",  # PyTorch container
    requirements=["transformers==4.33.0", "datasets==2.14.0", "pandas==2.0.3"],    # required pip packages
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest"
    # Using a standard PyTorch serving container for deployment (CPU in this example)
)

# Launch the training job on a GPU machine (e.g., 1 NVIDIA T4 GPU)
model = job.run(
    args=[],  # command-line args for trainer.py if any
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    base_output_dir=f"gs://{BUCKET_NAME}/flan_t5_medical",  # where to store model artifacts
)
# The job will fine-tune the model; after completion, `model` is a Vertex Model resource
print("Fine-tuned model artifact URI:", model.uri)
```

This Vertex job will:
*   Spin up the specified machine (with GPU) and run `trainer.py`.
*   Save the fine-tuned model to the Cloud Storage output directory and automatically register the model in Vertex AI (as we provided a serving container, Vertex knows how to handle the model artifact).
*   Return a model object we can use to deploy.

**Deploy the Model to an Endpoint:** Once fine-tuning is done, deploy the model to a Vertex endpoint for online predictions:

```python
# Deploy the fine-tuned model to a Vertex AI Endpoint for predictions
endpoint = model.deploy(
    machine_type="n1-standard-4",  # machine for serving
    min_replica_count=1,
    max_replica_count=1,
    traffic_split={"0": 100}      # all traffic to this deployed model
)
endpoint_uri = endpoint.resource_name
print("Deployed Vertex Endpoint:", endpoint_uri)
```

The model is now accessible via the endpoint for inference. (Under the hood, a service with the PyTorch serving container loads our model artifact and waits for prediction requests.)

## 4. Setting Up the Vector Database (Vertex AI Vector Search)

With the Q&A model ready, we set up the retrieval component: a vector store of domain knowledge that the system will search for relevant context. We will use Vertex AI Matching Engine (Vector Search) to store and query embeddings of documents.

**Document Corpus:** We need domain texts to populate the vector index. In our case, we can use the answers from the MedQuAD dataset as the knowledge base. Each answer (or part of an answer) will be a document chunk we can retrieve to assist in answering new questions. If you have additional documents (research articles, guidelines), you could include those as well.

**Chunking:** It's often beneficial to split long documents into smaller chunks (e.g., 200-300 words each) for more granular retrieval. For simplicity, we'll assume each Q&A answer is a reasonable chunk (if some answers are very long, they should be split by paragraphs or sentences). We’ll index each answer as a separate chunk, with the question or a snippet as metadata.

**Embedding Model:** Vertex AI offers powerful embedding models. We use the latest text embedding model (e.g., `text-embedding-005` or `textembedding-gecko@001` – dimension 768). This model converts text into a 768-dimensional vector in a semantic space. Let's generate embeddings for each document chunk and build the index:

```python
from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import os # Added import os

# Initialize Vertex AI (ensure you have logged in or running in a GCP environment)
PROJECT_ID = os.environ.get("PROJECT_ID") # Added PROJECT_ID definition
REGION = os.environ.get("REGION", "us-central1") # Added REGION definition
BUCKET_NAME = os.environ.get("BUCKET_NAME", f"{PROJECT_ID}-rag-bucket") # Added BUCKET_NAME definition
aiplatform.init(project=PROJECT_ID, location=REGION)
embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")  # or "text-embedding-005"

# Suppose we have a list of docs (the answers) and optional metadata like doc_id
docs = []       # list of answer texts
doc_ids = []    # corresponding IDs (e.g., "doc1", "doc2", etc.)
# (Fill docs and doc_ids by reading the dataset answers; omitted for brevity)

# Compute embeddings for all docs in batches
doc_embeddings = []
for i in range(0, len(docs), 100):
    batch = docs[i:i+100]
    # Corrected line: access .embeddings attribute (assuming list of objects with .values)
    embeddings_result = embed_model.get_embeddings(batch)
    doc_embeddings.extend([e.values for e in embeddings_result]) # Assuming result has .values

print(f"Computed {len(doc_embeddings)} embeddings of dimension {len(doc_embeddings[0]) if doc_embeddings else 0}.") # Added check for empty list
# Example: len(doc_embeddings) == len(docs), each embedding length == 768
```
*(Note: The original code had `embeddings = embed_model.get_embeddings(batch).embeddings`. The actual API might return a list of objects, each having a `.values` attribute for the vector. I've adjusted this based on common patterns, but the exact structure depends on the SDK version. Also added missing `PROJECT_ID`, `REGION`, `BUCKET_NAME` definitions and `import os`.)*

Now, create a Vertex AI Matching Engine Index to store these vectors. We use the Tree-AH (Approximate) index for scalable similarity search, configured for streaming updates so we can add data via API ([docs.llamaindex.ai](https://docs.llamaindex.ai/), [docs.llamaindex.ai](https://docs.llamaindex.ai/)):

```python
# Create a vector search index on Vertex AI
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="healthcare-doc-index",
    contents_delta_uri=f"gs://{BUCKET_NAME}/vector_index_content/", # Example path, might be needed
    dimensions=768,
    approximate_neighbors_count=10, # Example value, might be needed
    distance_measure_type="DOT_PRODUCT_DISTANCE",  # using dot-product for similarity
    shard_size="SHARD_SIZE_SMALL",                # for demonstration; use larger for >1M vectors
    index_update_method="STREAM_UPDATE"            # allow real-time updates via API
)
index.wait()  # wait for the index creation to complete (a few minutes)
print("Index resource name:", index.resource_name)
```
*(Note: Added potential required parameters like `contents_delta_uri` and `approximate_neighbors_count` based on typical Matching Engine usage. Removed the citation `[oaicite:8]{index=8}`.)*

Next, create an Index Endpoint and deploy the index to it so we can query it in real-time:

```python
# Create an index endpoint for querying
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="healthcare-index-endpoint",
    public_endpoint_enabled=True  # public endpoint for simplicity (no VPC)
)
index_endpoint.wait()
print("Index Endpoint resource:", index_endpoint.resource_name)

# Deploy the index to the endpoint
DEPLOYED_INDEX_ID = "healthcare_index_1"
index_endpoint.deploy_index(
    index=index,
    deployed_index_id=DEPLOYED_INDEX_ID,
    machine_type="e2-standard-2",  # type of machine hosting the index
    min_replica_count=1,
    max_replica_count=1
)
print(f"Deployed index with ID '{DEPLOYED_INDEX_ID}' to endpoint.")
```

Now we have a vector index ready to store vectors. We will upsert our document embeddings into the index. Each vector must have an ID. We’ll use our `doc_ids` (like `"Q1234"` or a hash of the question) as IDs. We can also attach metadata namespaces if we want to tag documents (not needed for basic use). Upserting can be done in batches:

```python
# Prepare datapoints for upsert: list of {"id": ..., "embedding": [...]}
datapoints = [
    aiplatform.MatchingEngineIndexDatapoint( # Use the specific class if available
        datapoint_id=doc_ids[i],
        feature_vector=doc_embeddings[i]
    )
    for i in range(len(doc_embeddings))
]

# Upsert datapoints into the index (in chunks if large)
# Note: The original text used index.upsert_datapoints, but the endpoint is usually used for upserting to a deployed index.
# Using index_endpoint.upsert_datapoints might be correct depending on the SDK version. Assuming index object works here.
CHUNK_SIZE = 100
for j in range(0, len(datapoints), CHUNK_SIZE):
    batch = datapoints[j:j+CHUNK_SIZE]
    index.upsert_datapoints(datapoints=batch) # Or index_endpoint.upsert_datapoints(...)
print("Inserted all document vectors into the index.")

```
*(Note: Adjusted the datapoint creation to potentially use `aiplatform.MatchingEngineIndexDatapoint` and noted the potential difference between upserting via the index object vs. the endpoint object.)*

This uses the streaming update capability of the index to add vectors programmatically. The index now contains all our medical answer embeddings and can return nearest neighbors for any query vector.

## 5. Building the RAG Serving Pipeline (Cloud Run Backend)

With the fine-tuned model and the vector index in place, the serving pipeline will handle user questions by retrieving relevant context and generating answers. We implement a simple backend (using Flask or FastAPI) to orchestrate the steps:

1.  Receive a user query (e.g., via a REST POST request).
2.  Embed the query using the same embedding model (so it’s in the same vector space as documents).
3.  **Vector Search:** Query the Matching Engine index for similar documents. This returns IDs of top-matching chunks.
4.  **Fetch Content:** Retrieve the actual text of those chunks (from a stored mapping or a database). Since our chunk texts are from the dataset, we might keep a dictionary in memory mapping `doc_id` -> text. For a more scalable solution, store documents in a database or Cloud Storage and retrieve by ID.
5.  **Construct Prompt:** Combine the retrieved context with the user question to form a prompt for the fine-tuned model. For instance:
    ```swift
    "Context: <retrieved text>\n\nQuestion: <user question>\nAnswer:"
    ```
    The fine-tuned model, having seen many Q&A examples during training, will use the context to produce a factual answer.
6.  **Generate Answer:** Call the Vertex AI Endpoint for the fine-tuned model with the prompt, and get the generated answer.
7.  **Return Answer:** Send the answer back as the response.

Below is an example Python (Flask) app implementing this. This code would run on Cloud Run to serve requests:

```python
# app.py: Flask app for query handling
import os, json
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

app = Flask(__name__)

# Initialize Vertex AI clients globally (reuse across requests for efficiency)
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
# Assuming endpoint and index endpoint details are available (e.g., via env vars or lookup)
FINETUNED_ENDPOINT_ID = os.environ.get("FINETUNED_ENDPOINT_ID") # e.g., projects/.../endpoints/...
VECTOR_INDEX_ENDPOINT_ID = os.environ.get("VECTOR_INDEX_ENDPOINT_ID") # e.g., projects/.../indexEndpoints/...
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "healthcare_index_1") # From step 4

aiplatform.init(project=PROJECT_ID, location=REGION)
embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
finetuned_endpoint = aiplatform.Endpoint(FINETUNED_ENDPOINT_ID)
vector_endpoint = aiplatform.MatchingEngineIndexEndpoint(VECTOR_INDEX_ENDPOINT_ID)

# Placeholder for mapping doc_id to text (load from dataset/storage)
doc_id_to_text = {} # Populate this map

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    user_question = data.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # 1. Embed the query
        query_embedding_result = embed_model.get_embeddings([user_question])
        query_vector = query_embedding_result[0].values # Access vector

        # 2. Vector Search
        NUM_NEIGHBORS = 3 # How many context chunks to retrieve
        search_response = vector_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vector],
            num_neighbors=NUM_NEIGHBORS
        )

        # 3. Fetch Content
        context_chunks = []
        if search_response and search_response[0]: # Check if neighbors found
            for neighbor in search_response[0]:
                doc_id = neighbor.id
                # Fetch text using the ID (replace with actual retrieval logic)
                chunk_text = doc_id_to_text.get(doc_id, f"Content for {doc_id} not found.")
                context_chunks.append(chunk_text)

        context = "\n".join(context_chunks)

        # 4. Construct Prompt
        prompt = f"Context: {context}\n\nQuestion: {user_question}\nAnswer:"

        # 5. Generate Answer (using the fine-tuned model endpoint)
        # Note: The actual prediction request format depends on the serving container used during deployment.
        # This is a generic example assuming a simple text in/out format.
        prediction_response = finetuned_endpoint.predict(instances=[{"prompt": prompt}]) # Adjust instance format as needed

        # Extract the generated answer (adjust based on actual response structure)
        generated_answer = prediction_response.predictions[0].get("content", "Could not generate answer.") # Example extraction

        # 6. Return Answer
        return jsonify({"answer": generated_answer})

    except Exception as e:
        print(f"Error processing query: {e}") # Log the error
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run locally for testing (Cloud Run uses Gunicorn or similar)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

```
*(Note: The original text was truncated. I've completed the Flask app based on the described steps, adding necessary initializations, error handling, and placeholders for environment variables and data loading. The exact prediction request/response format for the fine-tuned model and vector search depends heavily on the specific setup and SDK versions.)*

```bash
# (Continued from original text, likely instructions for deploying the Flask app)
# Build the container image (using Cloud Build)
# gcloud builds submit --tag gcr.io/$PROJECT_ID/rag-backend:latest .

# Deploy to Cloud Run
# gcloud run deploy rag-service --image gcr.io/$PROJECT_ID/rag-backend:latest \
#   --platform managed --region $REGION --allow-unauthenticated \
#   --set-env-vars=PROJECT_ID=$PROJECT_ID,REGION=$REGION,FINETUNED_ENDPOINT_ID=...,VECTOR_INDEX_ENDPOINT_ID=...,DEPLOYED_INDEX_ID=...
```
*(Note: Added hypothetical deployment commands based on the context.)*

This completes the setup for a domain-specific RAG Q&A system on Vertex AI using MedQuAD data. Further steps could involve adding more sophisticated chunking, evaluating performance, and integrating with a user interface.

