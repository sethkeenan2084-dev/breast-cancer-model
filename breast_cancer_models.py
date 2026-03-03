from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class ModelResult:
    """Stores evaluation results for one model."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray


class ModelRunner:
    """
    Trains and evaluates multiple classification models on the breast cancer dataset.

    Attributes
    ----------
    test_size : float
        Fraction of the dataset reserved for testing.
    random_state : int
        Seed for reproducibility.
    X_train, X_test, y_train, y_test
        Train/test splits populated by load_and_split().
    """

    def __init__(self, test_size: float = 0.30, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_split(self) -> None:
        """
        Loads the built-in breast cancer dataset and performs a train/test split.
        Uses stratification to preserve class balance in both splits.
        """
        data = load_breast_cancer()
        X, y = data.data, data.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

    def build_models(self) -> List[Tuple[str, Any]]:
        """
        Creates and returns the 3 classification models as (name, model) pairs.
        Pipelines are used for models that benefit from scaling.

        Returns
        -------
        list of (str, estimator)
        """
        # 1) Logistic Regression (scaled)
        log_reg = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=self.random_state))
        ])

        # 2) SVM with RBF kernel (scaled)
        svm_rbf = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=self.random_state))
        ])

        # 3) Random Forest (tree-based; scaling not required)
        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=self.random_state,
            class_weight="balanced"
        )

        return [
            ("Logistic Regression", log_reg),
            ("SVM (RBF Kernel)", svm_rbf),
            ("Random Forest", rf),
        ]

    def evaluate_model(self, name: str, model: Any) -> ModelResult:
        """
        Trains the model and evaluates it using multiple metrics.

        Notes
        -----
        - ROC-AUC uses predicted probabilities if available.
        - If a model does not support predict_proba, you'd typically use decision_function,
          but here all models do support probabilities (SVC uses probability=True).

        Returns
        -------
        ModelResult
        """
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Probabilities for ROC-AUC (positive class = 1)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_proba)
        cm = confusion_matrix(self.y_test, y_pred)

        return ModelResult(
            name=name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=auc,
            confusion_matrix=cm
        )

    def run(self) -> List[ModelResult]:
        """
        Full workflow: load/split data, build models, evaluate each model, and return results.
        """
        self.load_and_split()
        models = self.build_models()

        results = []
        for name, model in models:
            results.append(self.evaluate_model(name, model))

        return results


def print_results_table(results: List[ModelResult]) -> None:
    """
    Prints a clean comparison table and confusion matrices for each model.
    """
    # Sort by ROC-AUC then F1 (common “overall” ranking approach)
    results_sorted = sorted(results, key=lambda r: (r.roc_auc, r.f1), reverse=True)

    print("\n=== Model Performance (sorted by ROC-AUC, then F1) ===")
    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'ROC-AUC':>9}")
    print("-" * 64)
    for r in results_sorted:
        print(f"{r.name:<22} {r.accuracy:>7.4f} {r.precision:>7.4f} {r.recall:>7.4f} {r.f1:>7.4f} {r.roc_auc:>9.4f}")

    print("\n=== Confusion Matrices (rows=true, cols=pred) ===")
    for r in results_sorted:
        print(f"\n{r.name}")
        print(r.confusion_matrix)

    best = results_sorted[0]
    print("\n=== Best Model (by ROC-AUC then F1) ===")
    print(f"{best.name} | Acc={best.accuracy:.4f}, Prec={best.precision:.4f}, Recall={best.recall:.4f}, F1={best.f1:.4f}, ROC-AUC={best.roc_auc:.4f}")