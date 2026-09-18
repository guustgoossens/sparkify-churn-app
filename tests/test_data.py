from __future__ import annotations

import pandas as pd
import pytest

from churn_app.data import (
    LEAKY_PAGES,
    REQUIRED_COLUMNS,
    DataValidationError,
    build_user_profiles,
    drop_leaky_events,
    extract_state,
    filter_events,
    filter_to_window,
    filter_users,
    label_churn,
    load_events,
    validate_events,
)


class TestLoadEvents:
    def test_parquet_round_trip(self, raw_events, tmp_path):
        path = tmp_path / "events.parquet"
        raw_events.to_parquet(path)
        loaded = load_events(path)
        assert len(loaded) == len(raw_events)
        assert list(loaded.columns) == list(REQUIRED_COLUMNS)

    def test_csv_is_equivalent_to_parquet(self, raw_events, tmp_path):
        raw_events.to_parquet(tmp_path / "e.parquet")
        raw_events.to_csv(tmp_path / "e.csv", index=False)
        from_parquet = load_events(tmp_path / "e.parquet")
        from_csv = load_events(tmp_path / "e.csv")
        pd.testing.assert_frame_equal(
            from_parquet[["userId", "time", "page"]],
            from_csv[["userId", "time", "page"]],
            check_dtype=False,
        )

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_events(tmp_path / "nope.parquet")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "events.xlsx"
        path.write_text("not really excel")
        with pytest.raises(DataValidationError, match="Unsupported file type"):
            load_events(path)


class TestValidateEvents:
    def test_missing_columns_are_reported_by_name(self, raw_events):
        with pytest.raises(DataValidationError, match=r"\['page', 'userId'\]"):
            validate_events(raw_events.drop(columns=["userId", "page"]))

    def test_epoch_milliseconds_are_parsed(self, raw_events):
        raw_events["time"] = raw_events["time"].astype("int64") // 1_000_000
        events = validate_events(raw_events)
        assert events["time"].min() == pd.Timestamp("2018-10-01 10:00")

    def test_numeric_user_ids_become_strings(self, raw_events):
        raw_events["userId"] = raw_events["userId"].map({"u1": 1, "u2": 2, "u3": 3})
        events = validate_events(raw_events)
        assert set(events["userId"]) == {"1", "2", "3"}

    def test_rows_without_user_or_time_are_dropped(self, raw_events):
        raw_events.loc[0, "userId"] = ""
        raw_events.loc[1, "userId"] = None
        raw_events.loc[2, "time"] = pd.NaT
        assert len(validate_events(raw_events)) == len(raw_events) - 3

    def test_empty_log_is_rejected(self, raw_events):
        with pytest.raises(DataValidationError, match="no usable rows"):
            validate_events(raw_events.iloc[0:0])

    def test_output_is_sorted_and_input_untouched(self, raw_events):
        shuffled = raw_events.sample(frac=1, random_state=0)
        before = shuffled.copy()
        events = validate_events(shuffled)
        assert events.equals(
            events.sort_values(["userId", "time"]).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(shuffled, before)

    def test_extra_columns_are_ignored(self, raw_events):
        raw_events["firstName"] = "Ada"
        assert "firstName" not in validate_events(raw_events).columns


def test_extract_state():
    locations = pd.Series(
        ["Los Angeles, CA", "New York-Newark, NY-NJ-PA", None, "somewhere"]
    )
    assert extract_state(locations).tolist() == ["CA", "NY", "Unknown", "Unknown"]


def test_label_churn(events):
    assert label_churn(events).to_dict() == {"u1": True, "u2": False, "u3": False}


def test_build_user_profiles(events):
    profiles = build_user_profiles(events)
    assert profiles.index.tolist() == ["u1", "u2", "u3"]
    assert profiles["state"].tolist() == ["MA", "CA", "NY"]
    assert profiles["churned"].tolist() == [True, False, False]
    assert profiles.loc["u1", "n_events"] == 6
    assert profiles.loc["u1", "n_sessions"] == 2
    assert (profiles.loc["u3", "first_level"], profiles.loc["u3", "last_level"]) == (
        "free",
        "paid",
    )


class TestFilterUsers:
    @pytest.fixture
    def profiles(self, events):
        return build_user_profiles(events)

    def test_no_filter_keeps_everyone(self, profiles):
        assert len(filter_users(profiles)) == 3

    def test_level_uses_latest_subscription(self, profiles):
        assert filter_users(profiles, levels=["paid"]).index.tolist() == ["u1", "u3"]

    def test_filters_combine_with_and(self, profiles):
        kept = filter_users(profiles, levels=["paid"], genders=["F"], states=["NY"])
        assert kept.index.tolist() == ["u3"]

    def test_empty_selection_yields_no_users(self, profiles):
        assert filter_users(profiles, genders=[]).empty

    def test_min_events(self, profiles):
        assert filter_users(profiles, min_events=4).index.tolist() == ["u1"]


class TestFilterEvents:
    def test_no_filter_returns_everything(self, events):
        assert len(filter_events(events)) == len(events)

    def test_user_filter(self, events):
        assert set(filter_events(events, user_ids=["u2"])["userId"]) == {"u2"}

    def test_date_bounds_are_inclusive_of_whole_end_day(self, events):
        kept = filter_events(events, start="2018-10-02", end="2018-10-05")
        assert kept["time"].min() == pd.Timestamp("2018-10-02 08:00")
        assert kept["time"].max() == pd.Timestamp("2018-10-05 12:01")

    def test_exact_end_timestamp_is_respected(self, events):
        kept = filter_events(events, end="2018-10-01 10:05")
        assert len(kept) == 2

    def test_no_match_gives_empty_frame_with_columns(self, events):
        kept = filter_events(events, start="2030-01-01")
        assert kept.empty and list(kept.columns) == list(events.columns)


class TestFilterToWindow:
    def test_window_is_relative_to_each_users_first_event(self, events):
        kept = filter_to_window(events, window_days=7)
        # u1's day-10 session and u2's day-18 song fall outside their windows.
        assert kept.groupby("userId").size().to_dict() == {"u1": 3, "u2": 2, "u3": 3}

    def test_large_window_keeps_everything(self, events):
        assert len(filter_to_window(events, window_days=365)) == len(events)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_non_positive_window_is_rejected(self, events, bad):
        with pytest.raises(ValueError):
            filter_to_window(events, window_days=bad)


def test_drop_leaky_events(events):
    kept = drop_leaky_events(events)
    assert not kept["page"].isin(LEAKY_PAGES).any()
    assert len(kept) == len(events) - 2


class TestBundledSample:
    """Guards the data file that ships with the repository."""

    def test_loads_and_matches_documented_shape(self, sample_events):
        profiles = build_user_profiles(sample_events)
        assert len(profiles) == 400
        assert profiles["churned"].sum() == 89

    def test_contains_no_personal_name_columns(self, sample_events):
        assert not {"firstName", "lastName", "userAgent"} & set(sample_events.columns)
