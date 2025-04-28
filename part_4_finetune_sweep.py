import pandas as pd
import json
import time
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import torch # Ensure torch is imported if used as backend

# --- Check for required libraries ---
try:
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EvalPrediction,
    )
except ImportError:
    print("Error: transformers or datasets library not found.")
    print("Please install them: pip install transformers datasets accelerate")
    exit()

# --- Configuration ---
# W&B Configuration
WANDB_PROJECT = "pizza_request_extra_credit" # <<< CHANGE THIS TO YOUR PROJECT NAME

# Model Configuration
MODEL_NAME = "distilbert-base-uncased" # Smaller, faster fine-tuning than BERT

# Data Configuration
DATA_PATH = "pizza_request_dataset.json" # Path to the main dataset
TEXT_COLUMN = "request_text_edit_aware" # Text field to analyze
TARGET_COLUMN = "requester_received_pizza"
TEST_SPLIT_SIZE = 0.10 # Size of final test set (held out)
VALIDATION_SPLIT_SIZE = 0.10 # Size of validation set (from training set) for Trainer
RANDOM_STATE = 42 # Ensures consistent splitting

# Training Hyperparameters (Fixed)
BATCH_SIZE = 32 # As requested
NUM_EPOCHS = 10 # As requested (WILL BE VERY SLOW ON CPU)

# Hyperparameters to Sweep
LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4] # As requested
# LEARNING_RATES = [5e-5, 3e-5] # More typical fine-tuning LRs (add if time permits)

# --- Directories ---
OUTPUT_DIR_BASE = f"./{MODEL_NAME}-results"
LOGGING_DIR_BASE = f"./{MODEL_NAME}-logs"

# --- Load Data ---
def load_data(json_path):
    """Loads the main dataset from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Data loaded successfully from {json_path}.")
        print(f"Dataset shape: {df.shape}")
        # Ensure target is int, text is string
        if TARGET_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
            raise ValueError("Required columns not found")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').astype(str)
        # Rename target column to 'label' for Hugging Face compatibility
        df = df.rename(columns={TARGET_COLUMN: "label", TEXT_COLUMN: "text"})
        print(f"Target variable distribution: {df['label'].value_counts(normalize=True)}")
        return df[["text", "label"]]
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {json_path}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

# --- Tokenization --- #
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# --- Metrics Calculation --- #
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Apply sigmoid if logits are output (usually the case for binary classification)
    # or softmax if multi-class logits. Assuming binary logits here.
    # If model outputs probabilities directly, skip activation.
    # For robustness, let's check shape - typical classification output is (batch_size, num_classes)
    if len(preds.shape) == 2 and preds.shape[1] > 1:
         # Get probabilities for the positive class (class 1)
         scores = torch.softmax(torch.tensor(preds), dim=1)[:, 1].numpy()
         discrete_preds = np.argmax(preds, axis=1)
    else: # If it's already probabilities or a single logit
        # Apply sigmoid if it looks like logits (not strictly bounded between 0 and 1)
        if np.any(preds < 0) or np.any(preds > 1):
             scores = torch.sigmoid(torch.tensor(preds)).numpy().flatten()
        else:
             scores = preds.flatten()
        discrete_preds = (scores > 0.5).astype(int)

    labels = p.label_ids

    accuracy = accuracy_score(labels, discrete_preds)
    precision = precision_score(labels, discrete_preds, zero_division=0)
    recall = recall_score(labels, discrete_preds, zero_division=0)
    f1 = f1_score(labels, discrete_preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5 # Handle cases where only one class is present in labels/preds

    try:
        tn, fp, fn, tp = confusion_matrix(labels, discrete_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError: # Handle cases like only one class predicted
        specificity = 0 # Or decide how to handle based on context
        if (discrete_preds == 0).all():
            specificity = 1.0 if (labels == 0).any() else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': auc,
    }

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    df = load_data(DATA_PATH)

    # 2. Split Data (Train/Validation/Test)
    # First, separate the final test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )
    # Then split the remaining data into train and validation for the Trainer
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE), # Adjust size relative to train_val split
        random_state=RANDOM_STATE,
        stratify=train_val_df['label']
    )

    print(f"\nTrain size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    # Convert to Hugging Face Dataset objects
    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })

    # 3. Load Tokenizer
    print(f"\nLoading tokenizer for model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Tokenize Datasets
    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Remove original text column after tokenization to avoid issues with Trainer
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # 5. Hyperparameter Sweep Loop
    print(f"\n--- Starting Hyperparameter Sweep for Learning Rates: {LEARNING_RATES} ---")
    print(f"WARNING: Training {NUM_EPOCHS} epochs per run ON CPU will be VERY SLOW.")

    for lr in LEARNING_RATES:
        run_name = f"run_lr_{lr:.0e}" # e.g., run_lr_1e-05
        print(f"\n--- Starting W&B Run: {run_name} --- LEARN MORE AT https://wandb.ai/site ---")

        # Initialize W&B Run for each LR
        wandb.init(
            project=WANDB_PROJECT,
            config={
                "learning_rate": lr,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "model_name": MODEL_NAME,
            },
            name=run_name,
            reinit=True # Allows starting multiple runs in one script
        )

        # Load a fresh model for each run to avoid weight leakage
        print(f"Loading pre-trained model: {MODEL_NAME}...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        # Define Training Arguments for this run
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR_BASE}/{run_name}",
            logging_dir=f"{LOGGING_DIR_BASE}/{run_name}",
            learning_rate=lr,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01, # Common default
            evaluation_strategy="epoch", # Evaluate every epoch
            logging_strategy="epoch", # Log metrics every epoch
            save_strategy="no", # Don't save model checkpoints to save space/time
            # save_strategy="epoch", # Use if you want checkpoints
            # load_best_model_at_end=True, # Use if save_strategy is epoch
            report_to="wandb", # <<< Important: report metrics to W&B
            seed=RANDOM_STATE,
            fp16=False # Ensure FP16 is off for CPU
        )

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train the model
        print(f"Starting training for LR={lr}...")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        print(f"Training finished for LR={lr}. Duration: {end_time - start_time:.2f} seconds.")

        # Evaluate on final test set (optional, but good practice)
        # print("Evaluating on the final test set...")
        # test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        # print("Test Set Results:", test_results)
        # wandb.log({"test_" + k: v for k,v in test_results.items()})

        # Finish the W&B run for this LR
        wandb.finish()

    print("\n--- Hyperparameter Sweep Complete --- Check W&B Project for results ---") 