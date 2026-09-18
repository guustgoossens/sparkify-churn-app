"""Altair chart builders. They take tidy DataFrames and return charts."""

from __future__ import annotations

import altair as alt
import pandas as pd

# Colour follows the entity everywhere in the app: retained is always blue,
# churned always orange (pair checked for colour-vision-deficiency separation).
RETAINED, CHURNED = "#2a78d6", "#eb6834"
STATUS_DOMAIN = ["Retained", "Churned"]
STATUS_SCALE = alt.Scale(domain=STATUS_DOMAIN, range=[RETAINED, CHURNED])
MUTED, GRID = "#898781", "#e1e0d9"
# Daily bars span an explicit start -> end so their width scales with the axis.
# (Bars on a time-unit axis collapse to hairlines in narrow columns.) The 3h
# shortfall leaves a visible gap between neighbouring days.
DAY_BAR = pd.DateOffset(hours=21)


def style(chart: alt.Chart, height: int = 280) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID,
            domainColor="#c3c2b7",
            tickColor="#c3c2b7",
            labelColor=MUTED,
            titleColor="#52514e",
            titleFontWeight="normal",
        )
        .configure_legend(orient="top", title=None, labelColor="#52514e")
    )


def status_label(churned: pd.Series) -> pd.Series:
    return churned.map({True: "Churned", False: "Retained"})


def daily_active_users(events: pd.DataFrame) -> alt.Chart:
    daily = (
        events.assign(day=events["time"].dt.floor("D"))
        .groupby("day")["userId"]
        .nunique()
        .rename("active_users")
        .reset_index()
    )
    line = (
        alt.Chart(daily)
        .mark_line(color=RETAINED, strokeWidth=2)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("active_users:Q", title="Active users"),
            tooltip=[
                alt.Tooltip("day:T", title="Day", format="%a %d %b %Y"),
                alt.Tooltip("active_users:Q", title="Active users"),
            ],
        )
    )
    return style(line + _hover_points(line, "day"))


def daily_cancellations(events: pd.DataFrame, churn_page: str) -> alt.Chart:
    daily = (
        events.loc[events["page"] == churn_page]
        .assign(day=lambda d: d["time"].dt.floor("D"))
        .groupby("day")
        .size()
        .rename("cancellations")
        .reset_index()
        .assign(day_end=lambda d: d["day"] + DAY_BAR)
    )
    bars = (
        alt.Chart(daily)
        .mark_bar(color=CHURNED, orient="vertical")
        .encode(
            x=alt.X("day:T", title=None),
            x2="day_end:T",
            y=alt.Y(
                "cancellations:Q", title="Cancellations", axis=alt.Axis(tickMinStep=1)
            ),
            tooltip=[
                alt.Tooltip("day:T", title="Day", format="%a %d %b %Y"),
                alt.Tooltip("cancellations:Q", title="Cancellations"),
            ],
        )
    )
    return style(bars)


def churn_rate_by_segment(profiles: pd.DataFrame, segment: str, top_n: int = 12):
    rates = (
        profiles.groupby(segment)["churned"]
        .agg(users="size", churned="sum", churn_rate="mean")
        .reset_index()
        .sort_values("users", ascending=False)
        .head(top_n)
    )
    rates["label"] = (
        rates["churn_rate"].map("{:.0%}".format)
        + "  ("
        + rates["churned"].astype(str)
        + "/"
        + rates["users"].astype(str)
        + ")"
    )
    base = alt.Chart(rates).encode(
        y=alt.Y(f"{segment}:N", sort="-x", title=None),
        x=alt.X("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip(f"{segment}:N", title="Segment"),
            alt.Tooltip("users:Q", title="Users"),
            alt.Tooltip("churned:Q", title="Churned"),
            alt.Tooltip("churn_rate:Q", title="Churn rate", format=".1%"),
        ],
    )
    bars = base.mark_bar(
        color=CHURNED, cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=14
    )
    labels = base.mark_text(align="left", dx=6, color="#52514e").encode(text="label:N")
    return style(bars + labels, height=max(120, 28 * len(rates) + 40))


def feature_by_status(features: pd.DataFrame, feature: str, title: str) -> alt.Chart:
    data = features[[feature, "status"]].rename(columns={feature: "value"})
    box = (
        alt.Chart(data)
        .mark_boxplot(size=26, outliers={"size": 12, "opacity": 0.5})
        .encode(
            y=alt.Y("status:N", title=None, sort=STATUS_DOMAIN),
            x=alt.X("value:Q", title=title),
            color=alt.Color("status:N", scale=STATUS_SCALE, legend=None),
        )
    )
    return style(box, height=150)


def feature_importance(importances: pd.Series, labels: dict[str, str], top_n: int = 12):
    top = importances.head(top_n)
    data = pd.DataFrame(
        {"feature": top.index.map(labels), "importance": top.to_numpy()}
    )
    bars = (
        alt.Chart(data)
        .mark_bar(
            color=RETAINED, cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=14
        )
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X(
                "importance:Q", title="Relative importance", axis=alt.Axis(format="%")
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".1%"),
            ],
        )
    )
    return style(bars, height=28 * len(data) + 40)


def risk_distribution(scores: pd.DataFrame) -> alt.Chart:
    bars = (
        alt.Chart(scores)
        .mark_bar(opacity=0.85, binSpacing=2)
        .encode(
            x=alt.X(
                "risk_score:Q",
                bin=alt.Bin(step=0.1, extent=[0, 1]),
                title="Out-of-fold churn probability",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("count():Q", title="Users", stack=None),
            color=alt.Color("status:N", scale=STATUS_SCALE),
            row=alt.Row("status:N", title=None, sort=STATUS_DOMAIN, header=None),
            tooltip=[
                alt.Tooltip("status:N", title="Group"),
                alt.Tooltip("count():Q", title="Users"),
            ],
        )
        .resolve_scale(y="independent")
    )
    return style(bars, height=110)


def auc_by_window(results: pd.DataFrame) -> alt.Chart:
    auc_scale = alt.Scale(domain=[0.5, 1], zero=False, clamp=True)
    base = alt.Chart(results).encode(
        x=alt.X(
            "window:N",
            sort=None,
            title="Observation window",
            axis=alt.Axis(labelAngle=0),
        ),
    )
    error = base.mark_rule(color=MUTED).encode(
        y=alt.Y("low:Q", title="ROC AUC (5-fold)", scale=auc_scale), y2="high:Q"
    )
    points = base.mark_point(size=90, filled=True, opacity=1).encode(
        y=alt.Y("roc_auc:Q", title="ROC AUC (5-fold)", scale=auc_scale),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["Leak-free window", "Full history (leaky)"],
                range=[RETAINED, CHURNED],
            ),
        ),
        tooltip=[
            alt.Tooltip("window:N", title="Window"),
            alt.Tooltip("roc_auc:Q", title="ROC AUC", format=".3f"),
            alt.Tooltip("roc_auc_std:Q", title="Std across folds", format=".3f"),
        ],
    )
    return style(error + points, height=260)


def user_timeline(user_events: pd.DataFrame) -> alt.Chart:
    daily = (
        user_events.assign(
            day=user_events["time"].dt.floor("D"),
            kind=user_events["page"].where(
                user_events["page"] == "NextSong", "Other pages"
            ),
        )
        .replace({"kind": {"NextSong": "Songs"}})
        .groupby(["day", "kind"])
        .size()
        .rename("events")
        .reset_index()
        .assign(day_end=lambda d: d["day"] + DAY_BAR)
    )
    bars = (
        alt.Chart(daily)
        .mark_bar(orient="vertical")
        .encode(
            x=alt.X("day:T", title=None),
            x2="day_end:T",
            y=alt.Y("events:Q", title="Events per day"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Songs", "Other pages"], range=[RETAINED, "#1baf7a"]
                ),
            ),
            order=alt.Order("kind:N", sort="descending"),
            tooltip=[
                alt.Tooltip("day:T", title="Day", format="%a %d %b"),
                alt.Tooltip("kind:N", title="Type"),
                alt.Tooltip("events:Q", title="Events"),
            ],
        )
    )
    return style(bars, height=220)


def _hover_points(line: alt.Chart, field: str) -> alt.Chart:
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=[field], empty=False
    )
    return (
        line.mark_point(size=70, filled=True, color=RETAINED)
        .encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        .add_params(nearest)
    )
