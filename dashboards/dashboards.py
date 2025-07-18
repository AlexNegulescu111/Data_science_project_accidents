from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")
df["data_ora"] = pd.to_datetime(df["data_ora"])
df["ora"] = df["data_ora"].dt.hour
df["zi"] = df["data_ora"].dt.day_name()
df["luna"] = df["data_ora"].dt.month_name()

# Sidebar filter
st.sidebar.header("Filters")
judete_selectate = st.sidebar.multiselect("Select counties:", df["judet"].unique(), default=[])

if judete_selectate:
    df_filtrat = df[df["judet"].isin(judete_selectate)]
else:
    st.warning("Please select at least one county to view data.")
    df_filtrat = df.iloc[0:0]  # empty DataFrame

# Dashboard title
st.title("Traffic accident dashboard")

# KPIs
if not df_filtrat.empty:
    st.metric("Total accidents", len(df_filtrat))
    st.metric("Fatal accidents", df_filtrat["nr_morti"].gt(0).sum())
    st.metric("Total injured", df_filtrat["nr_raniti"].sum())

    # Accidents by hour
    st.subheader("Accidents by hour")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_filtrat, x="ora", color="steelblue", ax=ax1)
    st.pyplot(fig1)

    # Top 10 roads
    st.subheader("Top 10 roads with most accidents")
    top_drumuri = df_filtrat["drum"].value_counts().head(10)
    st.bar_chart(top_drumuri)

    # Heatmap day-hour
    st.subheader("Accidents heatmap (day x hour)")
    pivot = df_filtrat.groupby(["zi", "ora"]).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax2)
    st.pyplot(fig2)
