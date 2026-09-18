"""Loading, validating and filtering of the raw streaming event logs.

Every function here is pure (no Streamlit, no global state) so that it can be
unit-tested in isolation; the app only adds caching on top.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

CHURN_PAGE = "Cancellation Confirmation"
# Pages that are part of the cancellation flow itself. Using them as model
# inputs would leak the label (median 33s between "Cancel" and confirmation).
LEAKY_PAGES = ("Cancel", CHURN_PAGE)

REQUIRED_COLUMNS = (
    "userId",
    "sessionId",
    "time",
    "registration",
    "page",
    "level",
    "gender",
    "location",
    "length",
    "song",
    "artist",
)

SECONDS_PER_DAY = 86_400


class DataValidationError(ValueError):
    """Raised when an event log does not have the expected shape."""


def load_events(path: str | Path) -> pd.DataFrame:
    """Read an event log from a parquet or CSV file and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(path)
    elif suffix == ".csv":
        raw = pd.read_csv(path)
    else:
        raise DataValidationError(
            f"Unsupported file type '{suffix}' (expected .parquet or .csv)"
        )
    return validate_events(raw)


def validate_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and return a clean, chronologically sorted copy.

    - fails loudly when a required column is missing or the log is empty
    - parses ``time``/``registration`` (epoch milliseconds or datetimes)
    - drops events without a user id (logged-out traffic) or without timestamp
    - normalises ``userId`` to ``str`` so ids compare equal across file formats
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    events = raw.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in ("time", "registration"):
        events[column] = _to_datetime(events[column])

    events["userId"] = events["userId"].astype("string").str.strip()
    has_user = events["userId"].notna() & (events["userId"] != "")
    events = events[has_user & events["time"].notna()]
    if events.empty:
        raise DataValidationError("Event log contains no usable rows")

    events["userId"] = events["userId"].astype(str)
    return events.sort_values(["userId", "time"], kind="stable").reset_index(drop=True)


def _to_datetime(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def extract_state(location: pd.Series) -> pd.Series:
    """Map ``"Portland-Vancouver-Hillsboro, OR-WA"`` to ``"OR"``.

    Metro areas spanning several states are assigned to the first one listed.
    Missing or malformed locations become ``"Unknown"``.
    """
    state = location.astype("string").str.extract(r",\s*([A-Z]{2})", expand=False)
    return state.fillna("Unknown").astype(str)


def label_churn(events: pd.DataFrame) -> pd.Series:
    """Boolean churn label per user: did they reach the confirmation page?"""
    churned = events.loc[events["page"] == CHURN_PAGE, "userId"].unique()
    users = pd.Index(events["userId"].unique(), name="userId")
    return pd.Series(users.isin(churned), index=users, name="churned")


def build_user_profiles(events: pd.DataFrame) -> pd.DataFrame:
    """One row per user with the attributes used for filtering and display."""
    grouped = events.groupby("userId", sort=True)
    profiles = grouped.agg(
        gender=("gender", "first"),
        location=("location", "first"),
        first_level=("level", "first"),
        last_level=("level", "last"),
        registration=("registration", "first"),
        first_event=("time", "min"),
        last_event=("time", "max"),
        n_events=("page", "size"),
        n_sessions=("sessionId", "nunique"),
    )
    profiles["gender"] = profiles["gender"].fillna("Unknown")
    profiles["state"] = extract_state(profiles.pop("location"))
    profiles["churned"] = label_churn(events).reindex(profiles.index)
    return profiles


def filter_users(
    profiles: pd.DataFrame,
    levels: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
    states: Iterable[str] | None = None,
    min_events: int = 0,
) -> pd.DataFrame:
    """Subset user profiles. ``None`` means "no filter" for that attribute.

    An *empty* selection is a real filter and yields no users, which mirrors
    what a user expects after clearing a multiselect.
    """
    mask = profiles["n_events"] >= min_events
    if levels is not None:
        mask &= profiles["last_level"].isin(list(levels))
    if genders is not None:
        mask &= profiles["gender"].isin(list(genders))
    if states is not None:
        mask &= profiles["state"].isin(list(states))
    return profiles[mask]


def filter_events(
    events: pd.DataFrame,
    user_ids: Iterable[str] | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Keep events of the given users within ``[start, end]`` (both inclusive).

    A date-only ``end`` (midnight) is treated as the whole of that day.
    """
    mask = pd.Series(True, index=events.index)
    if user_ids is not None:
        mask &= events["userId"].isin(list(user_ids))
    if start is not None:
        mask &= events["time"] >= pd.Timestamp(start)
    if end is not None:
        end = pd.Timestamp(end)
        if end == end.normalize():
            mask &= events["time"] < end + pd.DateOffset(days=1)
        else:
            mask &= events["time"] <= end
    return events[mask]


def filter_to_window(events: pd.DataFrame, window_days: float) -> pd.DataFrame:
    """Keep only each user's first ``window_days`` days of activity.

    Observing every user for the same amount of time removes the temporal
    leakage of the raw logs, where churned users simply stop producing events.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    first_event = events.groupby("userId")["time"].transform("min")
    days_since_start = (events["time"] - first_event).dt.total_seconds()
    return events[days_since_start <= window_days * SECONDS_PER_DAY]


def drop_leaky_events(events: pd.DataFrame) -> pd.DataFrame:
    """Remove the cancellation-flow pages that would leak the churn label."""
    return events[~events["page"].isin(LEAKY_PAGES)]
