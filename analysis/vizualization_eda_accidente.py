"""EDA: distributions of accidents by time and location (figures in visuals/eda/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import seaborn as sns

from accidents.config import UNKNOWN_COUNTY
from accidents.data import load_clean
from accidents.plotting import plt, save_figure

SUBDIR = "eda"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def countplot(df, column, title, xlabel, name, order=None):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=df, hue=column, palette="Blues_d", order=order, legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Accidents")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    save_figure(name, subdir=SUBDIR)


def barh(values, title, xlabel, ylabel, name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=values.values, y=values.index, hue=values.index, palette="Blues_d", legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()
    save_figure(name, subdir=SUBDIR)


def main():
    df = load_clean()
    df.info()

    # 1. Distribution of traffic resumption time
    reluare = df["reluare"].dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(reluare, bins=30, kde=True, color="steelblue")
    plt.title("Traffic resumption time - distribution")
    plt.xlabel("Minutes")
    plt.ylabel("Accidents")
    plt.tight_layout()
    save_figure("resumption_histogram", subdir=SUBDIR)

    plt.figure(figsize=(6, 5))
    sns.boxplot(y=reluare[reluare > 0], orient="v")
    plt.yscale("log")
    plt.title("Traffic resumption time (log scale)")
    plt.ylabel("Minutes (log)")
    plt.grid(True, alpha=0.4)
    save_figure("resumption_boxplot", subdir=SUBDIR)

    # 2. Outliers via IQR
    q1, q3 = np.quantile(reluare, [0.25, 0.75])
    low_val, high_val = max(0, q1 - 1.5 * (q3 - q1)), q3 + 1.5 * (q3 - q1)
    inside = (reluare >= low_val) & (reluare <= high_val)
    print(f"Values between {low_val} - {high_val}: {inside.mean() * 100:.2f} % of the data")

    # 3. Time distributions
    df["hour"] = df["data_ora"].dt.hour
    df["day"] = df["data_ora"].dt.day_name()
    df["month"] = df["data_ora"].dt.month_name()
    df["year"] = df["data_ora"].dt.year

    countplot(df, "hour", "Accidents per hour", "Hour", "accidents_per_hour")
    countplot(df, "day", "Accidents per day of week", "Day", "accidents_per_day", order=DAY_ORDER)
    countplot(df, "month", "Accidents per month", "Month", "accidents_per_month", order=MONTH_ORDER)
    countplot(df, "year", "Accidents per year", "Year", "accidents_per_year")

    pivot = df.groupby(["day", "hour"]).size().unstack(fill_value=0).reindex(DAY_ORDER)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlOrRd")
    plt.title("Accidents per day and hour")
    plt.xlabel("Hour")
    plt.ylabel("Day")
    plt.tight_layout()
    save_figure("heatmap_day_hour", subdir=SUBDIR)

    # 4. Locations (by number of reported accidents - not a risk measure)
    known = df[df["judet"] != UNKNOWN_COUNTY]
    top_counties = known["judet"].value_counts().head(15)
    barh(top_counties, "Top 15 counties by number of reported accidents",
         "Accidents", "County", "top_counties")

    top_roads = df["drum"].dropna().value_counts().head(15)
    barh(top_roads, "Top 15 roads by number of reported accidents",
         "Accidents", "Road", "top_roads")

    # 5. Mean resumption time in the counties with most accidents (no outliers)
    subset = known[(known["reluare"] <= high_val) & known["judet"].isin(top_counties.index)]
    mean_resumption = subset.groupby("judet")["reluare"].mean().sort_values(ascending=False)
    barh(mean_resumption, "Mean traffic resumption time - top 15 counties (no outliers)",
         "Minutes", "County", "mean_resumption_by_county")


if __name__ == "__main__":
    main()
