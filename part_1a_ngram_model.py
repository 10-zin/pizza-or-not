import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --- Configuration ---
DATA_PATH = "pizza_request_dataset/pizza_request_dataset.json" # Assuming the file is in the same directory
TEXT_COLUMN = "request_text_edit_aware" # Use the edit-aware text field
TARGET_COLUMN = "requester_received_pizza"
TEST_SIZE = 0.10 # 10% for test set
RANDOM_STATE = 42 # Ensures consistent splitting across runs and models
N_FEATURES = 1000 # Top 500 unigrams + top 500 bigrams = 1000 features total
CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV

# --- Load Data ---
def load_data(json_path):
    """Loads the dataset from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Data loaded successfully from {json_path}.")
        print(f"Dataset shape: {df.shape}")
        # Basic check for required columns
        if TEXT_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
             raise ValueError(f"Required columns ('{TEXT_COLUMN}', '{TARGET_COLUMN}') not found.")
        # Convert target to integer (0 or 1)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        # Fill potential NaN values in text column
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
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
    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # Important for imbalanced datasets
    )
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
    print(f"Training target distribution: {y_train.value_counts(normalize=True)}")
    print(f"Testing target distribution: {y_test.value_counts(normalize=True)}")

    # 3. Feature Extraction (Top N-grams using TF-IDF)
    print(f"\nExtracting top {N_FEATURES} unigrams and bigrams using TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), # Unigrams and Bigrams
        max_features=N_FEATURES,
        stop_words='english' # Remove common English words
    )

    # Fit on training data ONLY, then transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF feature shapes: Train={X_train_tfidf.shape}, Test={X_test_tfidf.shape}")

    # 4. Define SVM and Parameter Grid
    svm_classifier = LinearSVC(random_state=RANDOM_STATE, dual=True, class_weight='balanced')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }

    # 5. Setup and Run GridSearchCV
    print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV to find best C for N-grams model...")
    grid_search = GridSearchCV(
        svm_classifier,    # The classifier instance
        param_grid,        # Parameters to tune
        scoring='roc_auc', # Optimize for AUC
        cv=CV_FOLDS,
        n_jobs=-1
    )

    # Fit GridSearchCV on the *transformed* training data
    grid_search.fit(X_train_tfidf, y_train)

    print(f"GridSearchCV training complete.")
    print(f"Best ROC AUC found during CV: {grid_search.best_score_:.4f}")
    print(f"Best C value found: {grid_search.best_params_['C']}")

    # Get the best SVM classifier found by the grid search
    best_svm_classifier = grid_search.best_estimator_

    # --- Feature Importance Analysis ---
    print("\n--- Top N-grams Contributing to Predictions ---")
    try:
        # Get coefficients and feature names
        coefficients = best_svm_classifier.coef_[0]
        feature_names = vectorizer.get_feature_names_out()

        # Create a DataFrame for easy sorting
        coef_df = pd.DataFrame({'ngram': feature_names, 'coefficient': coefficients})

        # Sort by coefficient value
        coef_df = coef_df.sort_values(by='coefficient', ascending=False)

        # Get top N positive (predicting success) and top N negative (predicting failure)
        N = 20 # Number of top features to show for each class
        top_positive_features = coef_df.head(N)
        top_negative_features = coef_df.tail(N).sort_values(by='coefficient', ascending=True)

        print(f"\nTop {N} n-grams predicting SUCCESS (Positive Coefficient):")
        for index, row in top_positive_features.iterrows():
            print(f"  {row['ngram']:<25} {row['coefficient']:.4f}")

        print(f"\nTop {N} n-grams predicting FAILURE (Negative Coefficient):")
        for index, row in top_negative_features.iterrows():
            print(f"  {row['ngram']:<25} {row['coefficient']:.4f}")

    except Exception as e:
        print(f"Could not perform feature importance analysis: {e}")
    # --- End Feature Importance Analysis ---

    # 6. Predict on Test Set using the Best Classifier
    print("\nPredicting on test set using the best model...")
    y_pred = best_svm_classifier.predict(X_test_tfidf)
    try:
        y_scores = best_svm_classifier.decision_function(X_test_tfidf)
    except AttributeError:
        # Fallback (shouldn't happen with LinearSVC)
        print("Warning: decision_function not available. Using predictions as scores.")
        y_scores = y_pred

    # 7. Evaluate Classifier on Test Set
    print("\nEvaluating best model on the test set...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_scores)
    except ValueError as e:
        print(f"Could not calculate AUC: {e}. Setting AUC to 0.")
        auc = 0.0

    # Calculate Specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        print("Could not compute full confusion matrix - likely predicts only one class.")
        specificity = 0

    # 8. Report Results
    print("\n--- Model 1 (Tuned C): N-grams (Top 1000 Uni/Bi-grams) ---")
    print(f"Best C found by GridSearchCV: {grid_search.best_params_['C']}")
    print(f"{ 'Metric':<12} {'Value':<10}")
    print("-" * 22)
    print(f"{ 'Accuracy':<12} {accuracy:<10.4f}")
    print(f"{ 'Precision':<12} {precision:<10.4f}")
    print(f"{ 'Recall':<12} {recall:<10.4f}")
    print(f"{ 'F1-Score':<12} {f1:<10.4f}")
    print(f"{ 'Specificity':<12} {specificity:<10.4f}")
    print(f"{ 'AUC':<12} {auc:<10.4f}")
    print("-" * 22) 