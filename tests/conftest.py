from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_app.data import validate_events

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "data" / "sample_events.parquet"


def make_event(user, time, page="NextSong", **overrides):
    is_song = page == "NextSong"
    event = {
        "userId": user,
        "sessionId": 1,
        "time": pd.Timestamp(time),
        "registration": pd.Timestamp("2018-09-01"),
        "page": page,
        "level": "free",
        "gender": "F",
        "location": "Boston-Cambridge-Newton, MA-NH",
        "length": 200.0 if is_song else np.nan,
        "song": "Song A" if is_song else None,
        "artist": "Artist A" if is_song else None,
    }
    event.update(overrides)
    return event


@pytest.fixture
def raw_events() -> pd.DataFrame:
    """Three hand-written users whose expected values are easy to derive.

    - u1: paid woman from MA, 2 sessions over 10 days, churns on Oct 11
    - u2: free man from CA, stays
    - u3: upgrades free -> paid, multi-state metro area, stays
    """
    rows = [
        # u1
        make_event("u1", "2018-10-01 10:00", level="paid"),
        make_event("u1", "2018-10-01 10:05", level="paid", song="Song B"),
        make_event("u1", "2018-10-01 10:06", "Thumbs Down", level="paid"),
        make_event("u1", "2018-10-11 09:00", "Roll Advert", level="paid", sessionId=2),
        make_event("u1", "2018-10-11 09:01", "Cancel", level="paid", sessionId=2),
        make_event(
            "u1",
            "2018-10-11 09:02",
            "Cancellation Confirmation",
            level="paid",
            sessionId=2,
        ),
        # u2
        make_event("u2", "2018-10-02 08:00", gender="M", location="Los Angeles, CA"),
        make_event(
            "u2",
            "2018-10-02 08:04",
            "Thumbs Up",
            gender="M",
            location="Los Angeles, CA",
        ),
        make_event("u2", "2018-10-20 08:00", gender="M", location="Los Angeles, CA"),
        # u3
        make_event("u3", "2018-10-05 12:00", location="New York-Newark, NY-NJ-PA"),
        make_event(
            "u3",
            "2018-10-05 12:01",
            "Submit Upgrade",
            location="New York-Newark, NY-NJ-PA",
        ),
        make_event(
            "u3",
            "2018-10-06 12:00",
            level="paid",
            artist="Artist B",
            location="New York-Newark, NY-NJ-PA",
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def events(raw_events) -> pd.DataFrame:
    return validate_events(raw_events)


@pytest.fixture(scope="session")
def sample_events() -> pd.DataFrame:
    from churn_app.data import load_events

    return load_events(SAMPLE_PATH)
