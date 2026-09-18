"""Streamlit front-end: explore and predict churn on streaming event logs.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Hosts that only install third-party requirements (e.g. Streamlit Community
# Cloud) do not install this project itself, so fall back to the source tree.
if not any((Path(entry) / "churn_app").is_dir() for entry in sys.path if entry):
    sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn_app import charts
from churn_app.data import (
    CHURN_PAGE,
    DataValidationError,
    build_user_profiles,
    filter_events,
    filter_users,
    load_events,
)
from churn_app.features import FEATURE_LABELS, build_user_features
from churn_app.model import MODEL_NAMES, ModelResult, evaluate_model

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "sample_events.parquet"
FULL_HISTORY = "Full history"
WINDOW_CHOICES = [3, 7, 14, 21, 30]

st.set_page_config(page_title="Churn explorer", page_icon="🎧", layout="wide")


# --- cached data access -------------------------------------------------------
# Cached functions are keyed on small hashable arguments (a path and the filter
# values) rather than on the 400k-row DataFrame itself, which is slow to hash.


@st.cache_data(show_spinner="Loading events…")
def get_events(path: str) -> pd.DataFrame:
    return load_events(path)


@st.cache_data(show_spinner=False)
def get_profiles(path: str) -> pd.DataFrame:
    return build_user_profiles(get_events(path))


@st.cache_data(show_spinner="Filtering events…")
def get_filtered_events(path: str, user_ids: tuple[str, ...], start, end):
    return filter_events(get_events(path), user_ids, start, end)


@st.cache_data(show_spinner="Computing user features…")
def get_features(path: str, user_ids: tuple[str, ...], start, end, window):
    return build_user_features(
        get_filtered_events(path, user_ids, start, end), window_days=window
    )


@st.cache_data(show_spinner="Cross-validating model…")
def get_model_result(
    path: str, user_ids: tuple[str, ...], start, end, window, model_name: str
) -> ModelResult:
    features = get_features(path, user_ids, start, end, window)
    churned = get_profiles(path)["churned"]
    return evaluate_model(features, churned, model_name)


def resolve_data_path(uploaded) -> str:
    """Uploaded file > CHURN_DATA_PATH environment variable > bundled sample."""
    if uploaded is None:
        return os.environ.get("CHURN_DATA_PATH", str(DEFAULT_DATA_PATH))
    content = uploaded.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / f"churn-{digest}{Path(uploaded.name).suffix}"
    if not target.exists():
        target.write_bytes(content)
    return str(target)


# --- sidebar ------------------------------------------------------------------

st.title("🎧 Churn explorer")
st.caption(
    "Who cancels their music-streaming subscription, what do they do before "
    "leaving, and how early can we tell? Built on user event logs."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Use your own event log",
        type=["parquet", "csv"],
        help="Same schema as the bundled sample. Leave empty to use the sample.",
    )
    try:
        data_path = resolve_data_path(uploaded)
        profiles = get_profiles(data_path)
    except (DataValidationError, FileNotFoundError) as error:
        st.error(f"Could not load the event log: {error}")
        st.stop()
    events = get_events(data_path)

    st.header("Filters")
    levels = st.multiselect(
        "Subscription (latest)", ["free", "paid"], default=["free", "paid"]
    )
    gender_options = sorted(profiles["gender"].unique())
    genders = st.multiselect("Gender", gender_options, default=gender_options)
    states = st.multiselect(
        "State",
        sorted(profiles["state"].unique()),
        placeholder="All states",
    )
    first_day, last_day = events["time"].min().date(), events["time"].max().date()
    period = st.date_input(
        "Period",
        value=(first_day, last_day),
        min_value=first_day,
        max_value=last_day,
    )
    min_events = st.slider("Minimum events per user", 0, 500, 0, step=10)

# st.date_input returns a 1-tuple while the user is still picking the end date.
start, end = (period[0], period[-1]) if period else (first_day, last_day)

selected = filter_users(profiles, levels, genders, states or None, min_events)
user_ids = tuple(selected.index)
view = get_filtered_events(data_path, user_ids, start, end)
if selected.empty or view.empty:
    st.warning("No users match these filters. Widen them in the sidebar.")
    st.stop()
selected = selected[selected.index.isin(view["userId"].unique())]

# --- headline numbers ---------------------------------------------------------

overall_rate = profiles["churned"].mean()
kpis = st.columns(4)
is_subset = len(selected) < len(profiles)
rate_gap = (selected["churned"].mean() - overall_rate) * 100
kpis[0].metric(
    "Users",
    f"{len(selected):,}",
    f"of {len(profiles):,}" if is_subset else None,
    delta_color="off",
    delta_arrow="off",
)
kpis[1].metric(
    "Churn rate",
    f"{selected['churned'].mean():.1%}",
    f"{rate_gap:+.1f} pts vs all users" if is_subset else None,
    delta_color="inverse",
)
kpis[2].metric("Events", f"{len(view):,}")
kpis[3].metric("Songs played", f"{(view['page'] == 'NextSong').sum():,}")

overview_tab, behaviour_tab, model_tab, user_tab = st.tabs(
    ["Overview", "Behaviour", "Prediction", "User explorer"]
)

# --- overview -----------------------------------------------------------------

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Daily active users")
        st.altair_chart(charts.daily_active_users(view), width="stretch")
    with right:
        st.subheader("Cancellations per day")
        st.altair_chart(charts.daily_cancellations(view, CHURN_PAGE), width="stretch")

    st.subheader("Churn rate by segment")
    segment_names = {
        "Subscription (latest)": "last_level",
        "Subscription (at first event)": "first_level",
        "Gender": "gender",
        "State (12 largest)": "state",
    }
    segment = st.radio("Segment by", list(segment_names), horizontal=True)
    st.altair_chart(
        charts.churn_rate_by_segment(selected, segment_names[segment]), width="stretch"
    )
    st.caption(
        "Labels show churned / total users. Small segments are noisy: a 50% rate "
        "on 4 users is not a finding."
    )

# --- behaviour ----------------------------------------------------------------

with behaviour_tab:
    st.subheader("How do churned users behave differently?")
    full_features = get_features(data_path, user_ids, start, end, None)
    labelled = full_features.assign(
        status=charts.status_label(selected["churned"].reindex(full_features.index))
    )

    medians = labelled.groupby("status")[list(FEATURE_LABELS)].median().T
    medians = medians.reindex(columns=charts.STATUS_DOMAIN)
    spread = full_features.std().replace(0, 1)
    medians["Gap (in std)"] = (medians["Churned"] - medians["Retained"]) / spread
    medians = medians.sort_values("Gap (in std)", key=abs, ascending=False)

    feature = st.selectbox(
        "Feature",
        medians.index,
        format_func=FEATURE_LABELS.get,
        help="Sorted by how strongly the two groups differ.",
    )
    st.altair_chart(
        charts.feature_by_status(labelled, feature, FEATURE_LABELS[feature]),
        width="stretch",
    )
    st.caption(
        "Computed on each user's whole history within the selected period. "
        "Volume features (events, songs…) are lower for churned users partly "
        "*because* they left early. The Prediction tab deals with that bias."
    )
    st.dataframe(
        medians.rename(index=FEATURE_LABELS).rename_axis("Median per user"),
        width="stretch",
        column_config={
            "Retained": st.column_config.NumberColumn(format="%.2f"),
            "Churned": st.column_config.NumberColumn(format="%.2f"),
            "Gap (in std)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

# --- prediction ---------------------------------------------------------------

with model_tab:
    st.subheader("How early can we predict churn?")
    st.markdown(
        "Churned users stop producing events, so features built on the **full "
        "history** quietly encode the answer (few events ⇒ churned). The honest "
        "setting observes every user for the same **first N days** and predicts "
        "whether they will cancel later."
    )
    controls = st.columns([1, 1, 2])
    model_name = controls[0].selectbox("Model", MODEL_NAMES, index=1)
    window_choice = controls[1].selectbox(
        "Observation window",
        [*WINDOW_CHOICES, FULL_HISTORY],
        index=2,
        format_func=lambda w: w if w == FULL_HISTORY else f"First {w} days",
    )
    window = None if window_choice == FULL_HISTORY else window_choice

    try:
        result = get_model_result(data_path, user_ids, start, end, window, model_name)
    except ValueError as error:
        st.warning(str(error))
        result = None

    if result is not None:
        metrics = st.columns(4)
        metrics[0].metric(
            "ROC AUC",
            f"{result.roc_auc:.3f}",
            f"± {result.roc_auc_std:.3f} across folds",
            delta_color="off",
            delta_arrow="off",
        )
        metrics[1].metric("F1 (churn class)", f"{result.f1:.3f}")
        metrics[2].metric("Accuracy", f"{result.accuracy:.1%}")
        metrics[3].metric("Always-'retained' baseline", f"{1 - result.churn_rate:.1%}")
        if window is None:
            st.warning(
                "Full history leaks the outcome through activity volume. Treat "
                "these scores as an upper bound, not as deployable performance."
            )

        left, right = st.columns(2)
        with left:
            st.markdown("**What the model relies on**")
            st.altair_chart(
                charts.feature_importance(result.importances, FEATURE_LABELS),
                width="stretch",
            )
        with right:
            st.markdown("**Risk scores by actual outcome**")
            scores = result.risk_scores.to_frame().assign(
                status=charts.status_label(
                    selected["churned"].reindex(result.risk_scores.index)
                )
            )
            st.altair_chart(charts.risk_distribution(scores), width="stretch")
            st.caption(
                "Each user is scored by a model that never saw them (5-fold "
                "out-of-fold predictions)."
            )

        if st.toggle("Compare all observation windows", value=False):
            rows = []
            for choice in [*WINDOW_CHOICES, FULL_HISTORY]:
                days = None if choice == FULL_HISTORY else choice
                r = get_model_result(data_path, user_ids, start, end, days, model_name)
                rows.append(
                    {
                        "window": choice if days is None else f"{choice} days",
                        "kind": "Full history (leaky)"
                        if days is None
                        else "Leak-free window",
                        "roc_auc": r.roc_auc,
                        "roc_auc_std": r.roc_auc_std,
                        "low": r.roc_auc - r.roc_auc_std,
                        "high": r.roc_auc + r.roc_auc_std,
                    }
                )
            st.altair_chart(charts.auc_by_window(pd.DataFrame(rows)), width="stretch")
            st.caption("Points are mean AUC across folds; bars are ± one std.")

# --- user explorer ------------------------------------------------------------

with user_tab:
    st.subheader("Look at one user")
    table = selected[["churned", "last_level", "gender", "state", "n_events"]].copy()
    if result is not None:
        table["risk_score"] = result.risk_scores.reindex(table.index)
        table = table.sort_values("risk_score", ascending=False)
    user_id = st.selectbox(
        "User (highest predicted risk first)" if result is not None else "User",
        table.index,
    )
    user = table.loc[user_id]
    user_events = view[view["userId"] == user_id]

    facts = st.columns(5)
    facts[0].metric("Outcome", "Churned" if user["churned"] else "Retained")
    facts[1].metric(
        "Predicted risk",
        "n/a" if pd.isna(user.get("risk_score")) else f"{user['risk_score']:.0%}",
    )
    facts[2].metric("Subscription", user["last_level"])
    facts[3].metric("Events", f"{len(user_events):,}")
    facts[4].metric("Sessions", user_events["sessionId"].nunique())

    st.altair_chart(charts.user_timeline(user_events), width="stretch")
    with st.expander("Page visits"):
        st.dataframe(
            user_events["page"].value_counts().rename("visits").rename_axis("page"),
            width="stretch",
        )
