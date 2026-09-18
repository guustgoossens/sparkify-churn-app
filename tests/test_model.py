from __future__ import annotations

import pytest

from churn_app.data import label_churn
from churn_app.features import build_user_features
from churn_app.model import MODEL_NAMES, evaluate_model, make_model


@pytest.fixture(scope="module")
def dataset(sample_events):
    return build_user_features(sample_events, window_days=14), label_churn(
        sample_events
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_evaluates(dataset, model_name):
    features, churned = dataset
    result = evaluate_model(features, churned, model_name, n_splits=3)
    assert result.n_users == len(features)
    assert 0.5 < result.roc_auc <= 1.0
    assert result.risk_scores.between(0, 1).all()
    assert result.risk_scores.index.equals(features.index)
    assert result.importances.sum() == pytest.approx(1.0)
    assert result.importances.is_monotonic_decreasing


def test_evaluation_is_reproducible(dataset):
    features, churned = dataset
    first = evaluate_model(features, churned, "Logistic regression", n_splits=3)
    second = evaluate_model(features, churned, "Logistic regression", n_splits=3)
    assert first.roc_auc == second.roc_auc


def test_too_few_churners_is_a_clear_error(dataset):
    features, churned = dataset
    retained_only = features[~churned.reindex(features.index)]
    with pytest.raises(ValueError, match="widen the filters"):
        evaluate_model(retained_only, churned)


def test_missing_labels_are_rejected(dataset):
    features, churned = dataset
    with pytest.raises(ValueError, match="needs a churn label"):
        evaluate_model(features, churned.iloc[:10])


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("Deep magic")
