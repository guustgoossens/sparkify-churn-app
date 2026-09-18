"""Cross-validated churn models on top of the user-level features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("Logistic regression", "Random forest", "Gradient boosting")


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    n_users: int
    churn_rate: float
    roc_auc: float
    roc_auc_std: float
    f1: float
    accuracy: float
    # Out-of-fold churn probability per user: every score comes from a model
    # that never saw that user, so it is safe to display next to the label.
    risk_scores: pd.Series
    importances: pd.Series


def make_model(model_name: str, seed: int = 42) -> ClassifierMixin:
    if model_name == "Logistic regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=seed
            ),
        )
    if model_name == "Random forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "Gradient boosting":
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed
        )
    raise ValueError(f"Unknown model '{model_name}', expected one of {MODEL_NAMES}")


def evaluate_model(
    features: pd.DataFrame,
    churned: pd.Series,
    model_name: str = "Random forest",
    n_splits: int = 5,
    seed: int = 42,
) -> ModelResult:
    """Stratified k-fold evaluation; also returns OOF scores and importances."""
    y = churned.reindex(features.index)
    if y.isna().any():
        raise ValueError("Every user in `features` needs a churn label")
    y = y.astype(int)
    if y.value_counts().reindex([0, 1], fill_value=0).min() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} churned and {n_splits} retained users "
            "to cross-validate; widen the filters."
        )

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = make_model(model_name, seed)
    proba = cross_val_predict(model, features, y, cv=folds, method="predict_proba")[
        :, 1
    ]
    predicted = (proba >= 0.5).astype(int)
    fold_aucs = [
        roc_auc_score(y.iloc[test], proba[test]) for _, test in folds.split(features, y)
    ]

    model.fit(features, y)
    return ModelResult(
        model_name=model_name,
        n_users=len(y),
        churn_rate=float(y.mean()),
        roc_auc=float(np.mean(fold_aucs)),
        roc_auc_std=float(np.std(fold_aucs)),
        f1=float(f1_score(y, predicted)),
        accuracy=float(accuracy_score(y, predicted)),
        risk_scores=pd.Series(proba, index=features.index, name="risk_score"),
        importances=_importances(model, features.columns),
    )


def _importances(model: ClassifierMixin, columns: pd.Index) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:  # scaled logistic regression: |coefficient| is comparable across features
        values = np.abs(model[-1].coef_[0])
    total = values.sum()
    values = values / total if total > 0 else values
    return pd.Series(values, index=columns, name="importance").sort_values(
        ascending=False
    )
