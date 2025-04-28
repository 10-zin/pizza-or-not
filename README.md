# pizza-or-not
Analysis on predicting likelihood of a successful pizza donation requests on r/Random_Acts_Of_Pizza posts.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/10-zin/pizza-or-not.git
    cd pizza-or-not
    ```
2.  **Create Conda Environment:**
    Ensure you have Anaconda or Miniconda installed.
    ```bash
    conda create --name soc-ass python=3.9  # Or your desired python version
    conda activate soc-ass
    pip install -r requirements.txt
    ```
## Outputs

The outputs from model runs are already saved in the `outputs/` directory:

- `1a.txt`: Results from the N-gram model showing feature importance and performance metrics
- `1b.txt`: Results from the Activity & Reputation model with feature importance analysis
- `1c.txt`: Results from the Combined model (N-grams + Activity features)
- `1d.txt`: Results from the Temporal Analysis

These files contain detailed information about model performance, including accuracy, precision, recall, F1-score, specificity, and AUC metrics, as well as feature importance analysis where applicable.

## Running the Code

Navigate to the specific part's directory and run the corresponding script or notebook:

*   **Part 1a: N-gram Model:**
    ```bash
    python part_1a_ngram_model.py 
    ```
*   **Part 1b: Activity & Reputation Features:**
    ```bash
    python part_1b_activity_model.py 
    ```
*   **Part 1c: Combined Model (N-grams + Activity):**
    ```bash
    python part_1c_combined_model.py 
    ```
*   **Part 1d: Temporal Analysis:**
    ```bash
    python part_1d_temporal_analysis.py 
    ```
*   **Part 4: Sentence Transformer Model:**
    ```bash
    python part_4_sentence_transformer.py 
    ```

The output files for each model will be saved in the `outputs/` directory.
