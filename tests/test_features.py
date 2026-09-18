from __future__ import annotations

import numpy as np
import pytest

from churn_app.features import FEATURE_LABELS, build_user_features


def test_one_row_per_user_with_documented_columns(events):
    features = build_user_features(events)
    assert features.index.tolist() == ["u1", "u2", "u3"]
    assert list(features.columns) == list(FEATURE_LABELS)
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_counts_for_hand_written_user(events):
    u1 = build_user_features(events).loc["u1"]
    assert u1["events"] == 4  # 6 raw events minus the 2 cancellation pages
    assert u1["sessions"] == 2
    assert u1["songs"] == 2
    assert u1["unique_songs"] == 2
    assert u1["thumbs_down"] == 1
    assert u1["ads"] == 1
    assert u1["active_days"] == 2
    assert u1["paid_share"] == 1.0
    assert u1["listening_hours"] == pytest.approx(400 / 3600)
    assert u1["account_age_days"] == pytest.approx(30 + 10 / 24)


def test_cancellation_pages_never_leak_into_features(events):
    with_cancel = build_user_features(events)
    without_cancel = build_user_features(
        events[~events["page"].isin(["Cancel", "Cancellation Confirmation"])]
    )
    assert with_cancel.equals(without_cancel)


def test_window_restricts_observation(events):
    full = build_user_features(events)
    windowed = build_user_features(events, window_days=7)
    assert windowed.loc["u1", "ads"] == 0 and full.loc["u1", "ads"] == 1
    assert windowed.loc["u2", "songs"] == 1 and full.loc["u2", "songs"] == 2
    assert (windowed["observed_days"] <= 7).all()


def test_upgrade_is_counted_and_paid_share_is_partial(events):
    u3 = build_user_features(events).loc["u3"]
    assert u3["upgrades"] == 1
    assert u3["paid_share"] == pytest.approx(1 / 3)


def test_empty_input_gives_empty_frame(events):
    features = build_user_features(events.iloc[0:0])
    assert features.empty
    assert list(features.columns) == list(FEATURE_LABELS)
