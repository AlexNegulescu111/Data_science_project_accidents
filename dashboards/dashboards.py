"""Streamlit dashboard: `streamlit run dashboards/dashboards.py`"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from accidents.data import load_processed

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_data
def get_data():
    df = load_processed()
    df["ora"] = df["data_ora"].dt.hour
    df["zi"] = df["data_ora"].dt.day_name()
    df["luna"] = df["data_ora"].dt.month_name()
    return df


df = get_data()

# Sidebar filter
st.sidebar.header("Filters")
judete_selectate = st.sidebar.multiselect("Select counties:", sorted(df["judet"].unique()), default=[])

if judete_selectate:
    df_filtrat = df[df["judet"].isin(judete_selectate)]
else:
    st.warning("Please select at least one county to view data.")
    df_filtrat = df.iloc[0:0]  # empty DataFrame

# Dashboard title
st.title("Traffic accident dashboard")

# KPIs
if not df_filtrat.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total accidents", len(df_filtrat))
    c2.metric("Fatal accidents", int(df_filtrat["nr_morti"].gt(0).sum()))
    c3.metric("Total injured", int(df_filtrat["nr_raniti"].sum()))

    # Accidents by hour
    st.subheader("Accidents by hour")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_filtrat, x="ora", color="steelblue", ax=ax1)
    st.pyplot(fig1)

    # Top 10 roads
    st.subheader("Top 10 roads with most accidents")
    st.bar_chart(df_filtrat["drum"].value_counts().head(10))

    # Heatmap day-hour
    st.subheader("Accidents heatmap (day x hour)")
    pivot = (df_filtrat.groupby(["zi", "ora"]).size().unstack(fill_value=0)
             .reindex(DAY_ORDER, fill_value=0))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax2)
    st.pyplot(fig2)
