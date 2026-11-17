"""Streamlit dashboard: `streamlit run dashboard/app.py`"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from accidents.config import UNKNOWN_COUNTY
from accidents.data import load_processed
from accidents.features import CIRCUMSTANCES, circumstance_flags, is_accident, road_type
from accidents.geo import plotly_county_map
from accidents.stats import wilson_ci

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ROAD_TYPES = {"A": "Motorway (A)", "DN": "National road (DN)", "DE": "Expressway (DE)",
              "DNCB": "Bucharest ring (DNCB)", "other": "Other", "unknown": "Not stated"}

st.set_page_config(page_title="Romanian traffic accidents", layout="wide")


@st.cache_data
def get_data():
    df = load_processed()
    df["is_accident"] = is_accident(df["description"])
    df["fatal"] = (df["deaths"] > 0).astype(int)
    df["road_type"] = df["road"].map(road_type).map(ROAD_TYPES)
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    flags = circumstance_flags(df["description"])
    return pd.concat([df, flags], axis=1)


df = get_data()

# ---------------------------------------------------------------- filters
st.sidebar.header("Filters")
first, last = df["timestamp"].min().date(), df["timestamp"].max().date()
# one isolated report from 2017; regular coverage starts in 2019
default_start = max(first, pd.Timestamp("2019-01-01").date())
period = st.sidebar.date_input("Period", value=(default_start, last), min_value=first, max_value=last)
counties = st.sidebar.multiselect(
    "Counties (empty = all)", sorted(c for c in df["county"].unique() if c != UNKNOWN_COUNTY))
road_types = st.sidebar.multiselect("Road type (empty = all)", list(ROAD_TYPES.values()))
accidents_only = st.sidebar.toggle(
    "Accidents only", value=True,
    help="Exclude reports that are not accidents (roadworks, vehicle fires, oversize transports...)")

mask = pd.Series(True, index=df.index)
if isinstance(period, tuple) and len(period) == 2:
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
    mask &= df["timestamp"].between(start, end, inclusive="left")
if counties:
    mask &= df["county"].isin(counties)
if road_types:
    mask &= df["road_type"].isin(road_types)
if accidents_only:
    mask &= df["is_accident"]
data = df[mask]

st.sidebar.caption(
    "Source: Romanian Police *Info trafic* reports (politiaromana.ro). Victims and vehicles are "
    "extracted automatically from the text and can contain errors.")

# ---------------------------------------------------------------- header + KPIs
st.title("Romanian traffic accidents")
if data.empty:
    st.warning("No reports match the selected filters.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reports" if not accidents_only else "Accidents", f"{len(data):,}")
k2.metric("Fatal accidents", f"{int(data['fatal'].sum()):,}")
k3.metric("Fatality rate", f"{data['fatal'].mean():.1%}",
          help="Share of accidents with at least one death")
k4.metric("Injured", f"{int(data['injured'].sum()):,}")
k5.metric("Median est. resumption", f"{data['resumption_min'].median():.0f} min"
          if data["resumption_min"].notna().any() else "-")

tab_map, tab_when, tab_where, tab_who, tab_reports = st.tabs(
    ["Map", "When", "Where", "Circumstances", "Reports"])

# ---------------------------------------------------------------- map
with tab_map:
    metric = st.radio("Colour counties by", ["Accidents", "Fatal accidents", "Fatality rate (%)"],
                      horizontal=True)
    by_county = (data[data["county"] != UNKNOWN_COUNTY].groupby("county")
                 .agg(accidents=("fatal", "size"), fatal=("fatal", "sum"), injured=("injured", "sum"))
                 .reset_index())
    by_county["rate"] = 100 * by_county["fatal"] / by_county["accidents"]
    column = {"Accidents": "accidents", "Fatal accidents": "fatal", "Fatality rate (%)": "rate"}[metric]
    hover = {r.county: (f"<b>{r.county}</b><br>accidents: {r.accidents}<br>fatal: {r.fatal}"
                        f"<br>fatality rate: {r.rate:.1f}%<br>injured: {r.injured}")
             for r in by_county.itertuples()}
    fig = plotly_county_map(by_county.set_index("county")[column].astype(float), hover=hover,
                            label=metric)
    st.plotly_chart(fig, width="stretch")
    if metric == "Fatality rate (%)":
        st.caption("Rates in counties with few accidents are unreliable - check the number of "
                   "accidents in the tooltip.")
    st.caption("Motorway reports often name no county; they are counted in the KPIs but not on the map.")

# ---------------------------------------------------------------- when
with tab_when:
    c1, c2 = st.columns(2)
    hourly = data.groupby("hour").size().reindex(range(24), fill_value=0)
    c1.plotly_chart(px.bar(x=hourly.index, y=hourly.values, labels={"x": "Hour", "y": "Accidents"},
                           title="By hour of day"), width="stretch")
    monthly = data.groupby("month").agg(accidents=("fatal", "size"), fatal=("fatal", "sum")).reset_index()
    c2.plotly_chart(px.line(monthly, x="month", y=["accidents", "fatal"], markers=True,
                            title="By month", labels={"value": "Count", "month": "", "variable": ""}),
                    width="stretch")
    pivot = data.groupby(["day", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=DAY_ORDER, columns=range(24), fill_value=0)
    st.plotly_chart(px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                              labels={"x": "Hour", "y": "", "color": "Accidents"},
                              title="Day of week x hour"), width="stretch")
    st.caption("Reports are published mostly between 06:00 and 22:00, so night hours are under-represented.")

# ---------------------------------------------------------------- where
with tab_where:
    roads = (data.dropna(subset=["road"]).groupby("road")
             .agg(accidents=("fatal", "size"), fatal_accidents=("fatal", "sum"),
                  deaths=("deaths", "sum"), injured=("injured", "sum"))
             .sort_values("accidents", ascending=False))
    roads["fatality_rate_%"] = (100 * roads["fatal_accidents"] / roads["accidents"]).round(1)
    min_n = st.slider("Minimum number of accidents per road", 1, 50, 10)
    st.dataframe(roads[roads["accidents"] >= min_n].head(30), width="stretch")

# ---------------------------------------------------------------- circumstances
with tab_who:
    rows = []
    for name in CIRCUMSTANCES:
        sub = data[data[f"has_{name}"] == 1]
        if len(sub) >= 10:
            low, high = wilson_ci(sub["fatal"].sum(), len(sub))
            rows.append({"circumstance": name.replace("_", " "), "accidents": len(sub),
                         "rate": sub["fatal"].mean(), "low": float(low), "high": float(high)})
    if rows:
        table = pd.DataFrame(rows).sort_values("rate")
        fig = px.scatter(table, x="rate", y="circumstance", hover_data=["accidents"],
                         error_x=table["high"] - table["rate"], error_x_minus=table["rate"] - table["low"],
                         labels={"rate": "Share of accidents that are fatal (95% CI)", "circumstance": ""},
                         title="How often an accident is fatal, by circumstance (at least 10 accidents)")
        fig.add_vline(x=data["fatal"].mean(), line_dash="dash", annotation_text="all")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough accidents in the selection.")

# ---------------------------------------------------------------- reports
with tab_reports:
    query = st.text_input("Search in descriptions", placeholder="e.g. pieton, DN 1, autobuz")
    shown = data
    if query:
        shown = shown[shown["description"].str.contains(query, case=False, regex=False, na=False)]
    cols = ["timestamp", "county", "road", "deaths", "injured", "resumption_min", "description"]
    st.caption(f"{len(shown):,} reports")
    st.dataframe(shown.sort_values("timestamp", ascending=False)[cols].head(500),
                 width="stretch", hide_index=True)
