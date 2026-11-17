"""Step 3: data/accidente_clean.csv -> data/accidente_processed.csv

Extracts the vehicles involved and the number of injured and dead from the
free-text description.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from accidents.config import CLEAN_CSV, PROCESSED_CSV
from accidents.text import (convert_to_number, extract_deaths,
                            extract_vehicles, extract_wounded)

VEHICLE_COLUMNS = ["veh_1", "nr_veh_1", "veh_2", "nr_veh_2", "veh_3", "nr_veh_3"]


def process(df):
    df = df.copy()
    vehicles = df["descriere"].apply(extract_vehicles)
    df[VEHICLE_COLUMNS] = pd.DataFrame(
        [(v1, n1, v2, n2, v3, n3) for (v1, n1), (v2, n2), (v3, n3) in vehicles],
        index=df.index,
    )
    df["nr_raniti"] = df["descriere"].apply(extract_wounded)
    df["nr_morti"] = df["descriere"].apply(extract_deaths)

    for col in ["nr_veh_1", "nr_veh_2", "nr_veh_3"]:
        df[col] = df[col].fillna("0").astype(str).apply(convert_to_number)
    df["total_nr_veh"] = df["nr_veh_1"] + df["nr_veh_2"] + df["nr_veh_3"]
    return df


def main():
    df = process(pd.read_csv(CLEAN_CSV))
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"{len(df)} rows, {int((df['nr_morti'] > 0).sum())} fatal -> {PROCESSED_CSV.name}")


if __name__ == "__main__":
    main()
