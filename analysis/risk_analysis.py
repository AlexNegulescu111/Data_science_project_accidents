"""Risk by road and county (figures in visuals/risk/).

Two complementary views, computed on actual accidents only:
  * burden   - number of fatal accidents (where most deaths happen; driven by
               traffic volume, which is not available in the data);
  * severity - share of accidents that are fatal, with a 95% Wilson interval.
               Ranking uses a shrunk rate, so a road with 1 fatal accident
               out of 2 does not top the list.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from accidents.config import UNKNOWN_COUNTY
from accidents.data import dangerous_accidents, load_processed
from accidents.features import is_accident
from accidents.geo import choropleth
from accidents.plotting import plt, save_figure
from accidents.stats import shrunk_rate, wilson_ci

SUBDIR = "risk"
MIN_ACCIDENTS = 30
TOP = 15


def severity_table(df, key):
    g = df.groupby(key).agg(accidents=("fatal", "size"), fatal=("fatal", "sum"),
                            deaths=("deaths", "sum"), injured=("injured", "sum"))
    overall = df["fatal"].mean()
    g["fatality_rate"] = g["fatal"] / g["accidents"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["fatal"], g["accidents"])
    g["shrunk_rate"] = shrunk_rate(g["fatal"], g["accidents"], overall)
    return g, overall


def plot_burden(table, label, name):
    top = table.sort_values("fatal", ascending=False).head(TOP).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top["fatal"], color="firebrick")
    for i, (f, n) in enumerate(zip(top["fatal"], top["accidents"])):
        ax.text(f, i, f" {int(f)} of {int(n)}", va="center")
    ax.set_xlabel("Fatal accidents")
    ax.set_title(f"Burden - {label} with the most fatal accidents")
    save_figure(name, fig=fig, subdir=SUBDIR)


def plot_severity(table, overall, label, name):
    eligible = table[table["accidents"] >= MIN_ACCIDENTS]
    top = eligible.sort_values("shrunk_rate", ascending=False).head(TOP).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    err = np.vstack([top["fatality_rate"] - top["ci_low"], top["ci_high"] - top["fatality_rate"]])
    ax.errorbar(top["fatality_rate"], top.index, xerr=err, fmt="o", color="firebrick", ecolor="gray", capsize=3)
    for i, (rate, n) in enumerate(zip(top["fatality_rate"], top["accidents"])):
        ax.annotate(f"n={int(n)}", (rate, i), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.axvline(overall, color="k", ls="--", lw=1, label=f"national average {overall:.1%}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of accidents that are fatal (95% confidence interval)")
    ax.set_title(f"Severity - {label} ranked by fatality rate (at least {MIN_ACCIDENTS} accidents)")
    ax.legend(loc="lower right")
    save_figure(name, fig=fig, subdir=SUBDIR)


def plot_county_maps(counties):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    choropleth(ax1, counties["fatal"].astype(float), cmap="Reds", label="fatal accidents")
    ax1.set_title("Burden - fatal accidents per county")
    choropleth(ax2, counties["shrunk_rate"] * 100, cmap="OrRd",
               label="fatal accidents per 100 accidents (shrunk)")
    ax2.set_title("Severity - share of accidents that are fatal")
    fig.suptitle("Romanian traffic accidents 2019-2025 (police Info-trafic reports)", fontsize=13)
    fig.tight_layout()
    save_figure("county_maps", fig=fig, subdir=SUBDIR)


def main():
    df = load_processed()
    df = df[is_accident(df["description"])].copy()
    df["fatal"] = (df["deaths"] > 0).astype(int)

    roads, overall = severity_table(df.dropna(subset=["road"]), "road")
    counties, _ = severity_table(df[df["county"] != UNKNOWN_COUNTY], "county")
    print(f"{len(df)} accidents, national fatality rate {overall:.1%}")
    cols = ["accidents", "fatal", "fatality_rate", "ci_low", "ci_high", "shrunk_rate"]
    print("\nRoads by severity:\n", roads[roads["accidents"] >= MIN_ACCIDENTS]
          .sort_values("shrunk_rate", ascending=False)[cols].head(10).round(3))
    print("\nCounties by severity:\n", counties[counties["accidents"] >= MIN_ACCIDENTS]
          .sort_values("shrunk_rate", ascending=False)[cols].head(10).round(3))

    plot_burden(roads, "roads", "roads_burden")
    plot_severity(roads, overall, "roads", "roads_severity")
    plot_burden(counties, "counties", "counties_burden")
    plot_severity(counties, overall, "counties", "counties_severity")
    plot_county_maps(counties)

    # Number of victims by traffic resumption interval (dangerous accidents)
    dangerous = dangerous_accidents(df).copy()
    dangerous["victims"] = dangerous["deaths"] + dangerous["injured"]
    dangerous["resumption_bin"] = pd.cut(dangerous["resumption_min"], bins=[0, 10, 20, 30, 60, 90, 180, 360, 9999],
                                      include_lowest=True)
    plt.figure(figsize=(10, 5))
    dangerous.groupby("resumption_bin", observed=False)["victims"].sum().plot(kind="bar", color="steelblue")
    plt.title("Dangerous accidents - victims by estimated traffic resumption interval")
    plt.xlabel("Traffic resumption interval (min)")
    plt.ylabel("Victims")
    plt.tight_layout()
    save_figure("victims_by_resumption", subdir=SUBDIR)


if __name__ == "__main__":
    main()
