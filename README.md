# Breast Cancer Classification Models

## Project Purpose
This project demonstrates how machine learning can support earlier and more accurate breast cancer detection by building and comparing multiple classification models using Scikit-Learn’s built-in Breast Cancer Wisconsin dataset. The goal is to train models using a standard train/test split, evaluate performance with multiple metrics, and determine which model performs best (or explain tradeoffs if performance is very close).

## What’s Included
- `breast_cancer_models.py`: Loads the dataset, splits into training/testing sets, trains 3 models, evaluates them, and prints a ranked comparison.
- Metrics reported:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix

## Standard Workflow Used
1. Load the dataset from `sklearn.datasets.load_breast_cancer()`.
2. Split the data into training and testing sets using `train_test_split()`.
   - Uses `stratify=y` to preserve the class balance.
   - Uses a fixed `random_state` for reproducibility.
3. Train each model using only the training set.
4. Evaluate each model on the held-out test set using the metrics above.
5. Select a “best” model primarily by ROC-AUC (overall discrimination), then by F1 (balance of precision/recall).

## Class Design and Implementation

### `ModelResult` (dataclass)
**Purpose:** Store the evaluation results for one model in a clean, structured way.

**Attributes:**
- `name`: Model name (string)
- `accuracy`: Accuracy score (float)
- `precision`: Precision score (float)
- `recall`: Recall score (float)
- `f1`: F1 score (float)
- `roc_auc`: ROC-AUC score (float)
- `confusion_matrix`: Confusion matrix (numpy array)

### `ModelRunner` (main controller class)
**Purpose:** Encapsulate the full ML workflow (load/split → build models → train/evaluate → return results).

**Attributes:**
- `test_size`: fraction reserved for the test set (default 0.30)
- `random_state`: seed for reproducibility (default 42)
- `X_train, X_test, y_train, y_test`: train/test splits (assigned in `load_and_split()`)

**Methods:**
- `load_and_split()`
  - Loads the dataset and performs a stratified train/test split.
- `build_models()`
  - Creates 3 classifiers:
    1) Logistic Regression (Pipeline + StandardScaler)
    2) SVM with RBF kernel (Pipeline + StandardScaler, probability enabled)
    3) Random Forest (no scaling required)
- `evaluate_model(name, model)`
  - Fits the model on the training set.
  - Predicts labels and probabilities for the test set.
  - Computes metrics and returns a `ModelResult`.
- `run()`
  - Runs the full pipeline and returns a list of `ModelResult` objects.

### Helper Functions
- `print_results_table(results)`
  - Prints a sorted comparison table and confusion matrices.
- `brief_decision_paragraph(results)`
  - Generates a short, submission-ready paragraph explaining which model performed best and why, using the actual metrics.

## How to Run
From your terminal:

```bash
python breast_cancer_models.py
