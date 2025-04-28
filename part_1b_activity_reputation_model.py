import pandas as pd
import json
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
DATA_PATH = "pizza_request_dataset/pizza_request_dataset.json" # Assuming the file is in the same directory
TARGET_COLUMN = "requester_received_pizza"
TEST_SIZE = 0.10 # 10% for test set
RANDOM_STATE = 42 # Ensures consistent splitting across runs and models
CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV

# --- Feature Lists ---
ACTIVITY_FEATURES_NUMERIC = [
    "requester_account_age_in_days_at_request",
    "requester_account_age_in_days_at_retrieval", # Corrected typo from user input
    "requester_days_since_first_post_on_raop_at_request",
    "requester_days_since_first_post_on_raop_at_retrieval",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_at_retrieval",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_comments_in_raop_at_retrieval",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_at_retrieval",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_posts_on_raop_at_retrieval",
    "requester_number_of_subreddits_at_request",
]
ACTIVITY_FEATURES_CATEGORICAL = [
    "post_was_edited" # Boolean, treated as categorical
    # Note: 'requester_subreddits_at_request' from the assignment list is excluded
    # because it contains lists and requires special handling, not suitable for this pipeline.
]

REPUTATION_FEATURES_NUMERIC = [
    "number_of_downvotes_of_request_at_retrieval",
    "number_of_upvotes_of_request_at_retrieval",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_minus_downvotes_at_retrieval",
    "requester_upvotes_plus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_retrieval",
]

ALL_NUMERIC_FEATURES = ACTIVITY_FEATURES_NUMERIC + REPUTATION_FEATURES_NUMERIC
ALL_CATEGORICAL_FEATURES = ACTIVITY_FEATURES_CATEGORICAL

# --- Load Data ---
def load_data(json_path):
    """Loads the dataset from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Data loaded successfully from {json_path}.")
        print(f"Dataset shape: {df.shape}")
        # Check for target column
        if TARGET_COLUMN not in df.columns:
             raise ValueError(f"Target column ('{TARGET_COLUMN}') not found.")
        # Check for feature columns (basic check)
        all_features = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
        missing_features = [col for col in all_features if col not in df.columns]
        if missing_features:
            print(f"Warning: The following specified features were not found in the dataframe: {missing_features}")
            if not (ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES):
                raise ValueError("Error: No specified features found in the dataset.")


        # Convert target to integer (0 or 1)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        # Convert boolean categorical feature to int
        if 'post_was_edited' in df.columns and 'post_was_edited' in ALL_CATEGORICAL_FEATURES:
             # Ensure boolean conversion only happens if the column exists and is used
             df['post_was_edited'] = df['post_was_edited'].astype(int)

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

    # Define Features (X) and Target (y)
    feature_columns = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
    if not feature_columns:
        print("Error: No features selected or available after checking. Exiting.")
        exit()
    print(f"\nUsing features: {feature_columns}")
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # 2. Split Data (Consistent Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # Important for imbalanced datasets
    )
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # 3. Feature Preprocessing Pipeline Setup
    print("\nSetting up preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ALL_NUMERIC_FEATURES),
            ('cat', categorical_transformer, ALL_CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    # 4. Create Full Pipeline with Preprocessor and Classifier Placeholder
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=RANDOM_STATE, dual=True, class_weight='balanced'))
    ])

    # 5. Define Parameter Grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100]
    }

    # 6. Setup and Run GridSearchCV
    print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV to find best C for Activity/Reputation model...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='roc_auc',
        cv=CV_FOLDS,
        n_jobs=-1
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    print(f"GridSearchCV training complete.")
    print(f"Best ROC AUC found during CV: {grid_search.best_score_:.4f}")
    print(f"Best C value found: {grid_search.best_params_['classifier__C']}")

    # Get the best estimator
    best_pipeline = grid_search.best_estimator_

    # --- Feature Importance Analysis ---
    print("\n--- Top Features Contributing to Predictions (Activity/Reputation Model) ---")
    try:
        # Get the classifier step from the best pipeline
        classifier = best_pipeline.named_steps['classifier']
        # Get the preprocessor step to access feature names
        preprocessor = best_pipeline.named_steps['preprocessor']

        # Get coefficients
        coefficients = classifier.coef_[0]

        # Get feature names generated by the preprocessor
        # This includes scaled numeric features and one-hot encoded categorical features
        feature_names = preprocessor.get_feature_names_out()

        if len(coefficients) == len(feature_names):
            coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
            coef_df = coef_df.sort_values(by='coefficient', ascending=False)

            N = 20 # Number of top features to show
            top_positive_features = coef_df.head(N)
            top_negative_features = coef_df.tail(N).sort_values(by='coefficient', ascending=True)

            print(f"\nTop {N} features predicting SUCCESS (Positive Coefficient):")
            for index, row in top_positive_features.iterrows():
                print(f"  {row['feature']:<60} {row['coefficient']:.4f}")

            print(f"\nTop {N} features predicting FAILURE (Negative Coefficient):")
            for index, row in top_negative_features.iterrows():
                print(f"  {row['feature']:<60} {row['coefficient']:.4f}")
        else:
             print(f"Could not align coefficients ({len(coefficients)}) with feature names ({len(feature_names)}). Skipping importance analysis.")

    except Exception as e:
        print(f"Could not perform feature importance analysis: {e}")
    # --- End Feature Importance Analysis ---

    # 7. Predict on Test Set using the Best Estimator
    print("\nPredicting on test set using the best model...")
    y_pred = best_pipeline.predict(X_test)
    try:
        y_scores = best_pipeline.decision_function(X_test)
    except AttributeError:
        print("Warning: decision_function not available. Trying predict_proba.")
        try:
            y_scores = best_pipeline.predict_proba(X_test)[:, 1]
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
    print("\n--- Model 2 (Tuned C): Activity & Reputation Features ---")
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