import pandas as pd
import json
import re
import os
from collections import defaultdict
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
MORAL_DIC_PATH = "resources/MoralFoundations.dic" # Path to the moral foundations dictionary
TEXT_COLUMN = "request_text_edit_aware" # Text field to analyze
TARGET_COLUMN = "requester_received_pizza"
TEST_SIZE = 0.10 # 10% for test set
RANDOM_STATE = 42 # Ensures consistent splitting across runs and models
CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV

# --- Define Moral Foundation Dimensions Required by Assignment ---
# Based on Appendix II and Part 1d description and verified .dic header
MORAL_DIMENSIONS = {
    "care_harm": [],
    "loyalty_betrayal": [],
    "authority_subversion": [],
    "sanctity_degradation": [],
    # Fairness/Cheating is excluded as per assignment Part 1d
}

# Map category IDs from MoralFoundations.dic to our required MORAL_DIMENSIONS keys
# Verified based on the actual MoralFoundations.dic header provided.
CATEGORY_ID_TO_DIMENSION_MAP = {
    # Care / Harm
    '1': 'care_harm',         # HarmVirtue
    '2': 'care_harm',         # HarmVice
    # Fairness / Cheating (IDs 3 & 4 are intentionally EXCLUDED)
    # '3': 'fairness_cheating', # FairnessVirtue
    # '4': 'fairness_cheating', # FairnessVice
    # Loyalty / Betrayal
    '5': 'loyalty_betrayal',  # IngroupVirtue
    '6': 'loyalty_betrayal',  # IngroupVice
    # Authority / Subversion
    '7': 'authority_subversion', # AuthorityVirtue
    '8': 'authority_subversion', # AuthorityVice
    # Sanctity / Degradation
    '9': 'sanctity_degradation', # PurityVirtue
    '10': 'sanctity_degradation', # PurityVice
    # MoralityGeneral (ID 11 is intentionally EXCLUDED as not specified in assignment)
}

# --- Hardcoded Index to Category Name Mapping ---
# Based on the user-provided list from MoralFoundations.dic header
IDX_TO_CATEGORY_NAME = {
    '01': 'HarmVirtue',
    '02': 'HarmVice',
    '03': 'FairnessVirtue',
    '04': 'FairnessVice',
    '05': 'IngroupVirtue',
    '06': 'IngroupVice',
    '07': 'AuthorityVirtue',
    '08': 'AuthorityVice',
    '09': 'PurityVirtue',
    '10': 'PurityVice',
    '11': 'MoralityGeneral'
}

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
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
        print(f"Target variable distribution: {df[TARGET_COLUMN].value_counts(normalize=True)}")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {json_path}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

# --- Load Moral Foundations Dictionary ---
def load_moral_foundations_dictionary(dic_path, category_map, required_dimensions):
    """Loads and parses the Moral Foundations dictionary (.dic format), skipping header."""
    moral_keywords = defaultdict(list)
    print(f"Loading Moral Foundations dictionary from: {dic_path}")
    line_num = 0 # Initialize line number for error reporting
    try:
        with open(dic_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # --- Skip header lines --- #
                if line_num < 14:
                    continue

                line = line.strip()
                if not line or line.startswith('%'): # Skip blank lines or potential separators later in file
                    continue

                # --- Process word lines --- #
                parts = line.split()
                if not parts:
                    continue

                word_pattern = parts[0]
                category_ids = parts[1:]

                if not category_ids:
                    print(f"  - Warning (Line {line_num}): No category ID found for word '{word_pattern}'. Skipping.")
                    continue

                for cat_id in category_ids:
                    clean_cat_id = ''.join(filter(str.isdigit, str(cat_id)))
                    if clean_cat_id:
                        dimension_key = category_map.get(clean_cat_id)
                        if dimension_key in required_dimensions:
                            moral_keywords[dimension_key].append(word_pattern)
                        # else: # Optional: Warn if a word's category isn't needed for the assignment
                            # if clean_cat_id in IDX_TO_CATEGORY_NAME:
                                # print(f"  - Info (Line {line_num}): Skipping word '{word_pattern}' for category {clean_cat_id} ({IDX_TO_CATEGORY_NAME[clean_cat_id]}) as it's not required.")
                            # else:
                                # print(f"  - Warning (Line {line_num}): Category ID {clean_cat_id} for word '{word_pattern}' not found in IDX_TO_CATEGORY_NAME.")
                    else:
                         print(f"  - Warning (Line {line_num}): Could not parse category ID '{cat_id}' for word '{word_pattern}'. Skipping.")

        print("Dictionary parsing complete (processed lines >= 14).")
        # Report counts for required dimensions
        final_keywords = {}
        dimensions_found = False
        for dim in required_dimensions:
            words = sorted(list(set(moral_keywords[dim]))) # Unique words per dimension
            if words:
                final_keywords[dim] = words
                print(f"  - Loaded {len(words)} unique word patterns for dimension '{dim}'")
                dimensions_found = True
            else:
                print(f"  - Warning: No words loaded for required dimension '{dim}'. Check CATEGORY_ID_TO_DIMENSION_MAP and .dic file format.")

        if not dimensions_found:
             print("\nERROR: No keywords loaded for ANY required moral dimension. \nCheck file path, format, and CATEGORY_ID_TO_DIMENSION_MAP. Cannot proceed.")
             exit()

        return final_keywords

    except FileNotFoundError:
        print(f"Error: Moral Foundations dictionary file not found at {dic_path}")
        exit()
    except Exception as e:
        print(f"An error occurred loading the moral dictionary (Line ~{line_num}): {e}")
        exit()

# --- Feature Extraction ---
def word_pattern_to_regex(pattern):
    """Converts a LIWC-style word pattern (like 'happ*') to a regex pattern."""
    if pattern.endswith('*'):
        # Match word boundary, the stem, and any following alphanumeric chars
        return r'\b' + re.escape(pattern[:-1]) + r'[a-zA-Z0-9]*\b'
    else:
        # Match exact word with boundaries
        return r'\b' + re.escape(pattern) + r'\b'

def extract_moral_features(text_series, moral_keywords_dict):
    """Calculates moral foundation features (ratio of keyword matches) for each text entry."""
    feature_df = pd.DataFrame(index=text_series.index)
    dimension_names = sorted(moral_keywords_dict.keys()) # Consistent column order
    print("Compiling regex patterns for moral dimensions...")
    regex_patterns = {}
    for name in dimension_names:
        # Combine all word patterns for the dimension into a single regex
        patterns_regex = '|'.join(word_pattern_to_regex(p) for p in moral_keywords_dict[name])
        if patterns_regex:
            regex_patterns[name] = re.compile(patterns_regex, re.IGNORECASE)
            print(f"  - Compiled regex for '{name}'")
        else:
             regex_patterns[name] = None # Handle cases where a dimension has no words
             print(f"  - No patterns to compile for '{name}'")

    print("Extracting moral foundation features...")
    total_texts = len(text_series)
    for i, text in enumerate(text_series):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total_texts} texts...")

        words = text.split() # Simple whitespace split
        total_word_count = len(words)

        for name in dimension_names:
            feature_col_name = f"moral_{name}_ratio"
            if total_word_count == 0 or regex_patterns[name] is None:
                feature_df.loc[text_series.index[i], feature_col_name] = 0.0
            else:
                matches = regex_patterns[name].findall(text)
                match_count = len(matches)
                feature_df.loc[text_series.index[i], feature_col_name] = match_count / total_word_count

    print(f"Moral foundation feature extraction complete. Shape: {feature_df.shape}")
    return feature_df

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data and Dictionary
    df = load_data(DATA_PATH)
    moral_keywords = load_moral_foundations_dictionary(
        MORAL_DIC_PATH,
        CATEGORY_ID_TO_DIMENSION_MAP,
        list(MORAL_DIMENSIONS.keys())
    )

    # 2. Split Data (Consistent Split)
    X_text = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Data split into training ({len(X_text_train)} texts) and testing ({len(X_text_test)} texts).")

    # 3. Extract Moral Features for Train and Test sets
    X_train_moral = extract_moral_features(X_text_train, moral_keywords)
    X_test_moral = extract_moral_features(X_text_test, moral_keywords)

    # 4. Create SVM Pipeline
    # Define the pipeline *without* the C parameter, as it will be set by GridSearchCV
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()), # Scale numeric moral features
        ('classifier', LinearSVC(random_state=RANDOM_STATE, dual=True, class_weight='balanced'))
    ])

    # 5. Define Parameter Grid for GridSearchCV
    param_grid = {
        # Access the 'C' parameter of the 'classifier' step in the pipeline
        'classifier__C': [0.01, 0.1, 1, 10, 100]
    }

    # 6. Setup and Run GridSearchCV
    print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV to find best C...")
    grid_search = GridSearchCV(
        pipeline,          # The pipeline object
        param_grid,        # Parameters to tune
        scoring='f1', # Optimize for AUC as it's good for imbalanced data
        cv=CV_FOLDS,       # Number of cross-validation folds
        n_jobs=-1          # Use all available CPU cores
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train_moral, y_train)

    print(f"GridSearchCV training complete.")
    print(f"Best ROC AUC found during CV: {grid_search.best_score_:.4f}")
    print(f"Best C value found: {grid_search.best_params_['classifier__C']}")

    # Get the best estimator (pipeline with best C)
    best_pipeline = grid_search.best_estimator_

    # --- Feature Importance Analysis ---
    print("\n--- Top Moral Foundation Features Contributing to Predictions ---")
    try:
        # Get the classifier step from the best pipeline
        classifier = best_pipeline.named_steps['classifier']
        # Get coefficients
        coefficients = classifier.coef_[0]

        # Feature names are the columns of the moral feature DataFrame
        # Ensure the order matches the data fed into the pipeline
        feature_names = X_train_moral.columns.tolist()

        if len(coefficients) == len(feature_names):
            coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
            coef_df = coef_df.sort_values(by='coefficient', ascending=False)

            # Since there are only 4 features, we can show all of them
            print("Moral Foundation Feature Coefficients (Positive = Predicts Success):")
            for index, row in coef_df.iterrows():
                print(f"  {row['feature']:<35} {row['coefficient']:.4f}")
        else:
             print(f"Could not align coefficients ({len(coefficients)}) with feature names ({len(feature_names)}). Skipping importance analysis.")

    except Exception as e:
        print(f"Could not perform feature importance analysis: {e}")
    # --- End Feature Importance Analysis ---

    # 7. Predict on Test Set using the Best Estimator
    print("\nPredicting on test set using the best model...")
    y_pred = best_pipeline.predict(X_test_moral)
    # Get decision function scores for AUC calculation
    try:
        y_scores = best_pipeline.decision_function(X_test_moral)
    except AttributeError:
        # This fallback shouldn't be needed for LinearSVC but is good practice
        print("Warning: decision_function not available. Trying predict_proba.")
        try:
            y_scores = best_pipeline.predict_proba(X_test_moral)[:, 1]
        except AttributeError:
            print("Warning: predict_proba also not available. Using predictions as scores.")
            y_scores = y_pred # Less ideal for AUC calculation

    # 8. Evaluate Classifier on Test Set
    print("\nEvaluating best model on the test set...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # Use the calculated y_scores for AUC
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
    print("\n--- Model 4 (Tuned C): Moral Foundations Features ---")
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