"""Risk scoring for roads and counties (figures in visuals/risk/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from accidents.data import dangerous_accidents, load_processed, model_frame
from accidents.plotting import plt, save_figure

SUBDIR = "risk"
WEIGHTS = {"acc_deaths": 0.4, "acc_wounded": 0.2, "acc_all": 0.2,
           "tot_deaths": 0.1, "tot_wounded": 0.1}


def risk_score(df, key):
    """Weighted risk score per `key` (road or county).

    Each component is the share of a national total that falls on that road/
    county. Missing components are 0 (not NaN), otherwise a road with no fatal
    accident would silently disappear from the ranking.
    """
    deaths = df[df["nr_morti"] > 0]
    wounded = df[df["nr_raniti"] > 0]
    parts = pd.concat({
        "acc_deaths": deaths[key].value_counts() / len(deaths),
        "acc_wounded": wounded[key].value_counts() / len(wounded),
        "acc_all": df[key].value_counts() / len(df),
        "tot_deaths": deaths.groupby(key)["nr_morti"].sum() / deaths["nr_morti"].sum(),
        "tot_wounded": wounded.groupby(key)["nr_raniti"].sum() / wounded["nr_raniti"].sum(),
    }, axis=1).fillna(0)
    risk = sum(parts[col] * w for col, w in WEIGHTS.items())
    return pd.DataFrame({"Real_risk": risk, "Z-score": (risk - risk.mean()) / risk.std()})


def plot_top(scores, title, xlabel, name):
    scores.sort_values("Real_risk", ascending=False).head(15).plot(
        kind="bar", figsize=(12, 6), width=0.8, colormap="Set2"
    )
    plt.title(title)
    plt.ylabel("risk value - real risk / z-score")
    plt.xlabel(xlabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Method")
    plt.tight_layout()
    save_figure(name, subdir=SUBDIR)


def main():
    df = model_frame(load_processed(), ["nr_morti", "nr_raniti", "reluare"])

    road_risk = risk_score(df, "drum")
    county_risk = risk_score(df, "judet")
    print("Top roads:\n", road_risk.sort_values("Real_risk", ascending=False).head(10))
    print("Top counties:\n", county_risk.sort_values("Real_risk", ascending=False).head(10))

    plot_top(road_risk, "Top 15 most dangerous roads", "Road", "top_roads_risk")
    plot_top(county_risk, "Top 15 most dangerous counties", "County", "top_counties_risk")

    # Number of victims by traffic resumption interval (dangerous accidents)
    dangerous = dangerous_accidents(df).copy()
    dangerous["victime"] = dangerous["nr_morti"] + dangerous["nr_raniti"]
    dangerous["reluare_bin"] = pd.cut(dangerous["reluare"], bins=[0, 10, 20, 30, 60, 90, 180, 360, 9999],
                                      include_lowest=True)
    plt.figure(figsize=(10, 5))
    dangerous.groupby("reluare_bin", observed=False)["victime"].sum().plot(kind="bar")
    plt.title("Dangerous accidents - victims by traffic resumption interval")
    plt.xlabel("Traffic resumption interval (min)")
    plt.ylabel("Victims")
    plt.tight_layout()
    save_figure("victims_by_resumption", subdir=SUBDIR)


if __name__ == "__main__":
    main()
