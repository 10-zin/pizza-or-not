import pandas as pd
import json
import re
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
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

# --- Configuration ---
DATA_PATH = "pizza_request_dataset/pizza_request_dataset.json" # Path to the main dataset
NARRATIVES_DIR_PATH = "pizza_request_dataset/narratives" # Directory containing narrative keyword files
TEXT_COLUMN = "request_text_edit_aware" # Text field to analyze
TARGET_COLUMN = "requester_received_pizza"
TEST_SIZE = 0.10 # 10% for test set
RANDOM_STATE = 42 # Ensures consistent splitting across runs and models
CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV

# --- Load Data ---
def load_data(json_path):
    """Loads the main dataset from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Data loaded successfully from {json_path}.")
        print(f"Dataset shape: {df.shape}")
        # Check for required columns
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

# --- Load Narrative Keywords ---
def load_narrative_keywords(dir_path):
    """Loads narrative keywords from text files in a directory."""
    narrative_keywords = {}
    try:
        print(f"Loading narrative keywords from: {dir_path}")
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                narrative_name = os.path.splitext(filename)[0]
                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        keywords = [line.strip() for line in f if line.strip()]
                    if keywords:
                        narrative_keywords[narrative_name] = keywords
                        print(f"  - Loaded {len(keywords)} keywords for '{narrative_name}'")
                    else:
                        print(f"  - Warning: No keywords found in '{filename}'")
                except Exception as e:
                    print(f"  - Error loading file '{filename}': {e}")
        if not narrative_keywords:
            print(f"Error: No keyword files found or loaded from {dir_path}. Exiting.")
            exit()
        return narrative_keywords
    except FileNotFoundError:
        print(f"Error: Narrative keywords directory not found at {dir_path}")
        exit()
    except Exception as e:
        print(f"An error occurred loading narrative keywords: {e}")
        exit()

# --- Feature Extraction ---
def extract_narrative_features(text_series, keywords_dict):
    """Calculates narrative features (ratio of keyword matches) for each text entry."""
    feature_df = pd.DataFrame(index=text_series.index)
    narrative_names = sorted(keywords_dict.keys()) # Ensure consistent column order
    print("Compiling regex patterns for narratives...")
    regex_patterns = {}
    for name in narrative_names:
        # Create a regex pattern: \b(word1|word2|...)\b for case-insensitive matching
        pattern = r'\b(' + '|'.join(re.escape(word) for word in keywords_dict[name]) + r')\b'
        regex_patterns[name] = re.compile(pattern, re.IGNORECASE)

    print("Extracting narrative features...")
    total_texts = len(text_series)
    for i, text in enumerate(text_series):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total_texts} texts...")

        words = text.split() # Simple whitespace split as per assignment description
        total_word_count = len(words)

        for name in narrative_names:
            feature_col_name = f"narrative_{name}_ratio"
            if total_word_count == 0:
                feature_df.loc[text_series.index[i], feature_col_name] = 0.0
                # print(f"  - {name}: 0 / {total_word_count} matches")
            else:
                matches = regex_patterns[name].findall(text)
                # feature_df.loc[text_series.index[i], f"matches_{name}"] = str(matches)
                match_count = len(matches)
                # print(f"  - {name}: {match_count} / {total_word_count} matches")
                feature_df.loc[text_series.index[i], feature_col_name] = match_count / total_word_count
        
        # feature_df.loc[text_series.index[i], "text"] = text

    print(f"Narrative feature extraction complete. Shape: {feature_df.shape}")
    return feature_df

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    df = load_data(DATA_PATH)
    narrative_keywords = load_narrative_keywords(NARRATIVES_DIR_PATH)

    # 2. Split Data (Consistent Split)
    # X contains the text for feature extraction, y is the target
    X_text = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Data split into training ({len(X_text_train)} texts) and testing ({len(X_text_test)} texts).")

    # 3. Extract Narrative Features
    X_train_narrative = extract_narrative_features(X_text_train, narrative_keywords)
    X_test_narrative = extract_narrative_features(X_text_test, narrative_keywords)

    # 

    # X_test_narrative.to_csv("X_test_narrative.csv", index=False)
    # exit()

    # 4. Create SVM Pipeline (Scaler + Classifier)
    print("Setting up SVM pipeline with StandardScaler...")
    svm_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()), # Scale numeric narrative features
        ('classifier', LinearSVC(random_state=RANDOM_STATE, dual=True, class_weight='balanced'))
    ])

    # 5. Define Parameter Grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100]
    }

    # 6. Setup and Run GridSearchCV
    print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV to find best C for Narratives model...")
    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid,
        scoring='roc_auc',
        cv=CV_FOLDS,
        n_jobs=-1
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train_narrative, y_train)

    print(f"GridSearchCV training complete.")
    print(f"Best ROC AUC found during CV: {grid_search.best_score_:.4f}")
    print(f"Best C value found: {grid_search.best_params_['classifier__C']}")

    # Get the best estimator
    best_pipeline = grid_search.best_estimator_

    # --- Feature Importance Analysis ---
    print("\n--- Top Narrative Features Contributing to Predictions ---")
    try:
        # Get the classifier step from the best pipeline
        classifier = best_pipeline.named_steps['classifier']
        # Get coefficients
        coefficients = classifier.coef_[0]

        # Feature names are the columns of the narrative feature DataFrame
        # Ensure the order matches the data fed into the pipeline
        feature_names = X_train_narrative.columns.tolist()

        if len(coefficients) == len(feature_names):
            coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
            coef_df = coef_df.sort_values(by='coefficient', ascending=False)

            # Since there are only 5 features, we can show all of them
            print("Narrative Feature Coefficients (Positive = Predicts Success):")
            for index, row in coef_df.iterrows():
                print(f"  {row['feature']:<30} {row['coefficient']:.4f}")
        else:
            print(f"Could not align coefficients ({len(coefficients)}) with feature names ({len(feature_names)}). Skipping importance analysis.")

    except Exception as e:
        print(f"Could not perform feature importance analysis: {e}")
    # --- End Feature Importance Analysis ---

    # 7. Predict on Test Set using the Best Estimator
    print("\nPredicting on test set using the best model...")
    y_pred = best_pipeline.predict(X_test_narrative)
    try:
        y_scores = best_pipeline.decision_function(X_test_narrative)
    except AttributeError:
        print("Warning: decision_function not available. Trying predict_proba.")
        try:
            y_scores = best_pipeline.predict_proba(X_test_narrative)[:, 1]
        except AttributeError:
            print("Warning: predict_proba also not available. Using predictions as scores.")
            y_scores = y_pred

    # 8. Evaluate Classifier on Test Set
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

    # 9. Report Results
    print("\n--- Model 3 (Tuned C): Narratives Features ---")
    print(f"Best C found by GridSearchCV: {grid_search.best_params_['classifier__C']}")
    print(f"{ 'Metric':<12} {'Value':<10}")
    print("-" * 22)
    print(f"{ 'Accuracy':<12} {accuracy:<10.4f}")
    print(f"{ 'Precision':<12} {precision:<10.4f}")
    print(f"{ 'Recall':<12} {recall:<10.4f}")
    print(f"{ 'F1-Score':<12} {f1:<10.4f}")
    print(f"{ 'Specificity':<12} {specificity:<10.4f}")
    print(f"{ 'AUC':<12} {auc:<10.4f}")
    print("-" * 22) 