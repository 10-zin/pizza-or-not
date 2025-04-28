import pandas as pd
import json
import time
import numpy as np # Add numpy import for diagnostics
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression # Replaced with XGBoost
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Check for sentence-transformers library ---
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers library not found.")
    print("Please install it: pip install sentence-transformers torch")
    exit()

# --- Check for xgboost library ---
try:
    from xgboost import XGBClassifier
except ImportError:
    print("Error: xgboost library not found.")
    print("Please install it: pip install xgboost")
    exit()

# --- Configuration ---
DATA_PATH = "pizza_request_dataset/pizza_request_dataset.json" # Path to the main dataset
TEXT_COLUMN = "request_text_edit_aware" # Text field to embed
TARGET_COLUMN = "requester_received_pizza"
TEST_SIZE = 0.10 # 10% for test set
RANDOM_STATE = 42 # Ensures consistent splitting across runs and models
CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV
# Model from sentence-transformers library - relatively lightweight and effective
# Other options: 'paraphrase-MiniLM-L3-v2', 'multi-qa-MiniLM-L6-cos-v1'
SENTENCE_MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1'

# --- Load Data ---
def load_data(json_path):
    """Loads the main dataset from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Data loaded successfully from {json_path}.")
        print(f"Dataset shape: {df.shape}")
        if TEXT_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
             raise ValueError(f"Required columns ('{TEXT_COLUMN}', '{TARGET_COLUMN}') not found.")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        # Convert text to string explicitly and fill NaNs
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').astype(str)
        print(f"Target variable distribution: {df[TARGET_COLUMN].value_counts(normalize=True)}")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {json_path}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    df = load_data(DATA_PATH)

    # 2. Split Data (Consistent Split)
    X_text = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"\nData split into training ({len(X_text_train)} texts) and testing ({len(X_text_test)} texts).")

    # 3. Load Sentence Transformer Model
    print(f"\nLoading sentence transformer model: {SENTENCE_MODEL_NAME}...")
    # Check device (use GPU if available, otherwise CPU)
    # Note: On CPU, embedding generation will be significantly slower.
    try:
        # Attempt to initialize with automatic device detection (GPU if available)
        model = SentenceTransformer(SENTENCE_MODEL_NAME)
        print(f"Model loaded successfully on device: {model.device}")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        print("Ensure 'sentence-transformers' and a backend (like 'torch') are installed.")
        exit()

    # 4. Generate Embeddings (This can take time on CPU)
    print("\nGenerating embeddings for training data...")
    start_time = time.time()
    # Convert Series to list for sentence-transformers
    X_train_embeddings = model.encode(X_text_train.tolist(), show_progress_bar=True)
    end_time = time.time()
    print(f"Training embedding generation took {end_time - start_time:.2f} seconds.")
    print(f"Shape of training embeddings: {X_train_embeddings.shape}")

    print("\nGenerating embeddings for test data...")
    start_time = time.time()
    X_test_embeddings = model.encode(X_text_test.tolist(), show_progress_bar=True)
    end_time = time.time()
    print(f"Test embedding generation took {end_time - start_time:.2f} seconds.")
    print(f"Shape of test embeddings: {X_test_embeddings.shape}")

    # 5. Calculate scale_pos_weight for XGBoost imbalance handling
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1
    print(f"\nCalculated scale_pos_weight for XGBoost: {scale_pos_weight_value:.2f}")

    # 6. Create Classification Pipeline with XGBoost
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()), # Keep scaler, often helps
        ('classifier', XGBClassifier(random_state=RANDOM_STATE,
                                     scale_pos_weight=scale_pos_weight_value,
                                     use_label_encoder=False,
                                     eval_metric='logloss')) # Use logloss or auc
    ])

    # 7. Define Parameter Grid for XGBoost
    # Reduced grid for faster CPU execution - expand if time permits
    param_grid = {
        'classifier__n_estimators': [100, 200], # Number of boosting rounds
        'classifier__max_depth': [3, 5],        # Max depth of a tree
        'classifier__learning_rate': [0.1]      # Step size shrinkage (fixed for simplicity here)
        # Add other params like 'subsample', 'colsample_bytree' for more tuning
    }

    # 8. Setup and Run GridSearchCV
    print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV to find best XGBoost parameters...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='roc_auc',
        cv=CV_FOLDS,
        n_jobs=-1 # Use available cores, but be mindful of memory
    )

    # Fit GridSearchCV on the training embeddings
    grid_search.fit(X_train_embeddings, y_train)

    print(f"GridSearchCV training complete.")
    print(f"Best ROC AUC found during CV: {grid_search.best_score_:.4f}")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Get the best estimator
    best_pipeline = grid_search.best_estimator_

    # 9. Predict on Test Set using the Best Estimator
    print("\nPredicting on test set using the best model...")
    y_pred = best_pipeline.predict(X_test_embeddings)
    try:
        y_scores = best_pipeline.predict_proba(X_test_embeddings)[:, 1]
    except AttributeError:
        print("Warning: predict_proba not available. Using predictions as scores.")
        y_scores = y_pred # Fallback

    # 10. Evaluate Classifier on Test Set
    print("\nEvaluating best model on the test set...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_scores)
    except ValueError as e:
        print(f"Could not calculate AUC: {e}. Setting AUC to 0.5.")
        auc = 0.5

    # Calculate Specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        print("Could not compute full confusion matrix.")
        specificity = 0

    # 11. Report Results
    print(f"\n--- Model 5 (Sentence Transformer: {SENTENCE_MODEL_NAME} + XGBoost Tuned) ---")
    print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
    print(f"{ 'Metric':<12} {'Value':<10}")
    print("-" * 22)
    print(f"{ 'Accuracy':<12} {accuracy:<10.4f}")
    print(f"{ 'Precision':<12} {precision:<10.4f}")
    print(f"{ 'Recall':<12} {recall:<10.4f}")
    print(f"{ 'F1-Score':<12} {f1:<10.4f}")
    print(f"{ 'Specificity':<12} {specificity:<10.4f}")
    print(f"{ 'AUC':<12} {auc:<10.4f}")
    print("-" * 22) 