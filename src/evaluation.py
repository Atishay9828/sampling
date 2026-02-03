"""Training and evaluation utilities."""

from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline


def run_experiments(X, y, samplers: Dict, models: Dict,
                    test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> pd.DataFrame:
    """Run experiments over provided samplers and models.

    Returns a pandas DataFrame with columns: Sampler, Model, Accuracy, F1, ROC_AUC.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )

    results = []

    for sampler_name, sampler in samplers.items():
        for model_name, model in models.items():
            steps = []
            if model_name in ["LogisticRegression", "SVM"]:
                steps.append(("scaler", StandardScaler()))

            if sampler is not None:
                steps.append(("sampler", sampler))

            steps.append(("model", model))
            pipe = Pipeline(steps)

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred

            results.append({
                "Sampler": sampler_name,
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC_AUC": roc_auc_score(y_test, y_prob)
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Accuracy", ascending=False)
    return df


def save_results(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
