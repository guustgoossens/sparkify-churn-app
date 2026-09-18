from __future__ import annotations

import pytest

from churn_app import charts
from churn_app.data import CHURN_PAGE


@pytest.mark.parametrize(
    "build",
    [
        lambda events: charts.daily_cancellations(events, CHURN_PAGE),
        lambda events: charts.user_timeline(events[events["userId"] == "u1"]),
    ],
    ids=["cancellations", "user_timeline"],
)
def test_daily_bars_have_explicit_width(events, build):
    """Regression: bars on a bare time axis shrank to hairlines in narrow columns.

    Each daily bar must span day -> day_end and stay vertical.
    """
    spec = build(events).to_dict()
    assert spec["encoding"]["x2"]["field"] == "day_end"
    assert spec["mark"]["orient"] == "vertical"
    rows = next(iter(spec["datasets"].values()))
    assert all(row["day_end"] > row["day"] for row in rows)
