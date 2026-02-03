"""Model definitions used in experiments."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models():
    """Return a mapping of model name -> sklearn estimator."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(class_weight="balanced", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier()
    }
