"""Risk by vehicle type (figures in visuals/vehicles/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seaborn as sns

from accidents.config import UNKNOWN_COUNTY
from accidents.data import dangerous_accidents, load_processed, normalize_vehicles
from accidents.plotting import plt, save_figure

SUBDIR = "vehicles"
MIN_ACCIDENTS = 10


def main():
    df = load_processed()
    df = df[df["judet"] != UNKNOWN_COUNTY].dropna(subset=["veh_1"])
    df = normalize_vehicles(df)

    # 1. Vehicles involved in dangerous accidents (fatal or >= 2 injured)
    dangerous = dangerous_accidents(df)
    involved = None
    for i in (1, 2, 3):
        s = dangerous.groupby(f"veh_{i}")[f"nr_veh_{i}"].sum()
        involved = s if involved is None else involved.add(s, fill_value=0)
    involved = involved.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=involved.values, y=involved.index, color="firebrick")
    for i, v in enumerate(involved.values):
        ax.text(v, i, f" {int(v)}", va="center", ha="left")
    plt.title("Vehicles involved in serious accidents")
    plt.xlabel("Number of vehicles")
    plt.ylabel("Type of vehicle")
    plt.grid(True, axis="x", linestyle="dashed")
    plt.tight_layout()
    save_figure("vehicles_in_serious_accidents", subdir=SUBDIR)

    # 2. Mean number of deaths per accident in which a vehicle type is involved
    total_deaths = total_count = None
    for i in (1, 2, 3):
        s = df.groupby(f"veh_{i}")["nr_morti"].sum()
        c = df[f"veh_{i}"].value_counts()
        total_deaths = s if total_deaths is None else total_deaths.add(s, fill_value=0)
        total_count = c if total_count is None else total_count.add(c, fill_value=0)

    mask = total_count > MIN_ACCIDENTS
    mean_deaths = (total_deaths / total_count)[mask].sort_values(ascending=False)
    counts = total_count[mask]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=mean_deaths.values, y=mean_deaths.index, color="gray")
    for i, key in enumerate(mean_deaths.index):
        ax.text(mean_deaths[key] + 0.002, i, f"{mean_deaths[key]:.3f} ({int(counts[key])} acc.)",
                va="center", ha="left")
    plt.title(f"Mean number of deaths per accident, by vehicle type involved (> {MIN_ACCIDENTS} accidents)")
    plt.xlabel("Mean deaths per accident")
    plt.ylabel("Type of vehicle")
    plt.grid(True, axis="x", linestyle="dashed")
    plt.tight_layout()
    save_figure("mean_deaths_by_vehicle", subdir=SUBDIR)


if __name__ == "__main__":
    main()
