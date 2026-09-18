"""End-to-end smoke tests: run the real Streamlit script headlessly."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import REPO_ROOT


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)


def test_app_renders_on_bundled_sample(app):
    app.run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Users"] == "400"
    assert metrics["Churn rate"] == "22.2%"
    assert "ROC AUC" in metrics


def test_filters_narrow_the_population(app):
    app.run()
    app.sidebar.multiselect[0].set_value(["paid"]).run()
    assert not app.exception
    users = next(m for m in app.metric if m.label == "Users")
    assert 0 < int(users.value) < 400


def test_window_comparison_runs_every_window(app):
    app.run()
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert any("mean AUC across folds" in c.value for c in app.caption)


def test_impossible_filter_shows_a_warning_instead_of_crashing(app):
    app.run()
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No users match" in w.value for w in app.warning)


def test_missing_data_file_is_reported(app, monkeypatch):
    monkeypatch.setenv("CHURN_DATA_PATH", "/does/not/exist.parquet")
    app.run()
    assert not app.exception
    assert any("Could not load" in e.value for e in app.error)
