"""Risk by vehicle type / road user involved (figures in visuals/vehicles/).

Each accident counts once per vehicle type it involves (an accident between
two cars counts once for "autoturism"), and the death toll is not split
between vehicles - the data does not say whose occupant died. The question
answered is therefore: "when an accident involves X, how often is it fatal?"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from accidents.config import VEHICLE_LABELS_EN
from accidents.data import dangerous_accidents, load_processed, normalize_vehicles
from accidents.features import circumstance_flags, is_accident
from accidents.plotting import plt, save_figure
from accidents.stats import wilson_ci

SUBDIR = "vehicles"
MIN_ACCIDENTS = 15


def involvement(df):
    """One 0/1 column per vehicle type / vulnerable road user."""
    vehicles = pd.get_dummies(df[["vehicle_1", "vehicle_2", "vehicle_3"]].stack()).groupby(level=0).max()
    flags = circumstance_flags(df["description"])
    vehicles["pedestrian"] = flags["has_pedestrian"]
    vehicles["bicicleta"] = np.maximum(vehicles.get("bicicleta", 0), flags["has_cyclist"])
    vehicles = vehicles.reindex(df.index, fill_value=0).astype(int)
    return vehicles.rename(columns=VEHICLE_LABELS_EN)


def main():
    df = load_processed()
    df = normalize_vehicles(df[is_accident(df["description"])])
    fatal = (df["deaths"] > 0).astype(int)
    overall = fatal.mean()
    inv = involvement(df)

    table = pd.DataFrame({
        "accidents": inv.sum(),
        "fatal": inv.mul(fatal, axis=0).sum(),
        "serious": inv.loc[dangerous_accidents(df).index].sum(),
    })
    table = table[table["accidents"] >= MIN_ACCIDENTS]
    table["fatality_rate"] = table["fatal"] / table["accidents"]
    table["ci_low"], table["ci_high"] = wilson_ci(table["fatal"], table["accidents"])
    print(table.sort_values("fatality_rate", ascending=False).round(3))

    # 1. Involvement in serious accidents (fatal or >= 2 injured)
    serious = table["serious"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(serious.index, serious.values, color="firebrick")
    for i, v in enumerate(serious.values):
        ax.text(v, i, f" {int(v)}", va="center")
    ax.set_xlabel("Serious accidents involving this vehicle type")
    ax.set_title("Vehicle types involved in serious accidents (fatal or 2+ injured)")
    save_figure("vehicles_in_serious_accidents", fig=fig, subdir=SUBDIR)

    # 2. Fatality rate with 95% confidence interval
    t = table.sort_values("fatality_rate")
    fig, ax = plt.subplots(figsize=(9, 6))
    err = np.vstack([t["fatality_rate"] - t["ci_low"], t["ci_high"] - t["fatality_rate"]])
    ax.errorbar(t["fatality_rate"], t.index, xerr=err, fmt="o", color="firebrick", ecolor="gray", capsize=3)
    for i, (rate, n) in enumerate(zip(t["fatality_rate"], t["accidents"])):
        ax.annotate(f"n={int(n)}", (rate, i), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.axvline(overall, color="k", ls="--", lw=1, label=f"all accidents {overall:.1%}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of accidents that are fatal (95% confidence interval)")
    ax.set_title(f"How often an accident is fatal, by vehicle / road user involved (n >= {MIN_ACCIDENTS})")
    ax.legend(loc="lower right")
    save_figure("fatality_rate_by_vehicle", fig=fig, subdir=SUBDIR)


if __name__ == "__main__":
    main()
