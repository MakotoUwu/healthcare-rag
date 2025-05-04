"""
Fine-tuning script for the T5 model on healthcare QA data.
This script is designed to be run as a custom training job on Vertex AI.
"""
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch
from datasets import Dataset

# Hyperparameters and paths
MODEL_NAME = "google/flan-t5-base"
DATA_PATH = os.environ.get("DATA_PATH", "medquad_qa.csv")  # CSV with 'question','answer'
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "model_output")
EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 512

def load_and_prepare_data(data_path):
    """Load dataset and prepare it for training."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare training data in seq2seq format
    train_data = []
    for q, a in zip(df["question"], df["answer"]):
        prompt = f"Question: {q}\nAnswer:"  # prompt format for model
        train_data.append({"prompt": prompt, "answer": a})
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    print(f"Prepared dataset with {len(train_dataset)} examples")
    
    return train_dataset

def preprocess_function(examples, tokenizer):
    """Tokenize inputs and targets."""
    inputs = tokenizer(examples["prompt"], max_length=MAX_LENGTH, truncation=True, padding="max_length")
    outputs = tokenizer(examples["answer"], max_length=MAX_LENGTH, truncation=True, padding="max_length")
    
    # Replace pad token id in labels to -100 so they are ignored in loss
    outputs["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels]
        for labels in outputs["input_ids"]
    ]
    
    examples["input_ids"] = inputs["input_ids"]
    examples["attention_mask"] = inputs["attention_mask"]
    examples["labels"] = outputs["input_ids"]
    
    return examples

def main():
    """Main function to execute the training pipeline."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading {MODEL_NAME} model and tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Load and prepare the dataset
    train_dataset = load_and_prepare_data(DATA_PATH)
    
    # Tokenize the dataset
    print("Tokenizing dataset")
    tokenized_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["prompt", "answer"]
    )
    
    # Set up Trainer
    print("Setting up trainer")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=f"{OUTPUT_DIR}/logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    print("Starting training")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
