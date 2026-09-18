"""User-level behavioural features computed from event logs."""

from __future__ import annotations

import pandas as pd

from churn_app.data import SECONDS_PER_DAY, drop_leaky_events, filter_to_window

# page name -> feature name
PAGE_COUNTS = {
    "NextSong": "songs",
    "Thumbs Up": "thumbs_up",
    "Thumbs Down": "thumbs_down",
    "Add to Playlist": "add_to_playlist",
    "Add Friend": "add_friend",
    "Roll Advert": "ads",
    "Error": "errors",
    "Help": "help_visits",
    "Home": "home_visits",
    "Settings": "settings_visits",
    "Downgrade": "downgrade_visits",
    "Upgrade": "upgrade_visits",
    "Submit Downgrade": "downgrades",
    "Submit Upgrade": "upgrades",
}

FEATURE_LABELS = {
    "events": "Events",
    "sessions": "Sessions",
    "songs": "Songs played",
    "unique_songs": "Unique songs",
    "unique_artists": "Unique artists",
    "thumbs_up": "Thumbs up",
    "thumbs_down": "Thumbs down",
    "add_to_playlist": "Playlist adds",
    "add_friend": "Friends added",
    "ads": "Adverts heard",
    "errors": "Errors",
    "help_visits": "Help page visits",
    "home_visits": "Home page visits",
    "settings_visits": "Settings page visits",
    "downgrade_visits": "Downgrade page visits",
    "upgrade_visits": "Upgrade page visits",
    "downgrades": "Downgrades submitted",
    "upgrades": "Upgrades submitted",
    "listening_hours": "Listening time (h)",
    "observed_days": "Days observed",
    "active_days": "Active days",
    "events_per_day": "Events per day",
    "songs_per_day": "Songs per day",
    "sessions_per_day": "Sessions per day",
    "songs_per_session": "Songs per session",
    "thumbs_down_ratio": "Thumbs-down ratio",
    "ads_per_song": "Adverts per song",
    "error_rate": "Error rate",
    "artist_diversity": "Artist diversity",
    "paid_share": "Share of events on paid tier",
    "account_age_days": "Account age at first event (days)",
}


def build_user_features(
    events: pd.DataFrame, window_days: float | None = None
) -> pd.DataFrame:
    """Aggregate events into one feature row per user.

    Cancellation-flow pages are always excluded. When ``window_days`` is given,
    only each user's first ``window_days`` days are used (leak-free setting);
    otherwise the whole log is used, which is informative for exploration but
    optimistic for modelling.
    """
    events = drop_leaky_events(events)
    if window_days is not None:
        events = filter_to_window(events, window_days)
    if events.empty:
        return pd.DataFrame(columns=list(FEATURE_LABELS)).rename_axis("userId")

    grouped = events.groupby("userId", sort=True)
    features = grouped.agg(
        events=("page", "size"),
        sessions=("sessionId", "nunique"),
        unique_songs=("song", "nunique"),
        unique_artists=("artist", "nunique"),
        listening_seconds=("length", "sum"),
        first_event=("time", "min"),
        last_event=("time", "max"),
        registration=("registration", "first"),
    )

    page_counts = (
        events[events["page"].isin(PAGE_COUNTS)]
        .groupby(["userId", "page"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(PAGE_COUNTS), fill_value=0)
        .rename(columns=PAGE_COUNTS)
    )
    features = features.join(page_counts).fillna({c: 0 for c in page_counts})
    features[list(PAGE_COUNTS.values())] = features[list(PAGE_COUNTS.values())].astype(
        int
    )

    features["listening_hours"] = features.pop("listening_seconds") / 3600
    features["active_days"] = (
        events.assign(day=events["time"].dt.floor("D"))
        .groupby("userId")["day"]
        .nunique()
    )
    features["paid_share"] = (
        events.assign(paid=events["level"].eq("paid")).groupby("userId")["paid"].mean()
    )

    span = (features["last_event"] - features["first_event"]).dt.total_seconds()
    # At least ~2.4h so single-session users do not get absurd per-day rates.
    features["observed_days"] = (span / SECONDS_PER_DAY).clip(lower=0.1)
    account_age = (
        features["first_event"] - features["registration"]
    ).dt.total_seconds()
    features["account_age_days"] = (account_age / SECONDS_PER_DAY).clip(lower=0)
    features = features.drop(columns=["first_event", "last_event", "registration"])

    features["events_per_day"] = features["events"] / features["observed_days"]
    features["songs_per_day"] = features["songs"] / features["observed_days"]
    features["sessions_per_day"] = features["sessions"] / features["observed_days"]
    features["songs_per_session"] = features["songs"] / features["sessions"]
    thumbs = features["thumbs_up"] + features["thumbs_down"]
    features["thumbs_down_ratio"] = features["thumbs_down"] / (thumbs + 1)
    features["ads_per_song"] = features["ads"] / (features["songs"] + 1)
    features["error_rate"] = features["errors"] / features["events"]
    features["artist_diversity"] = features["unique_artists"] / (features["songs"] + 1)

    return features[list(FEATURE_LABELS)].fillna(0.0)
