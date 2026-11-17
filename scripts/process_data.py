"""Step 3: data/accidents_clean.csv -> data/accidents_processed.csv

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

VEHICLE_COLUMNS = ["vehicle_1", "vehicle_1_count", "vehicle_2", "vehicle_2_count", "vehicle_3", "vehicle_3_count"]


def process(df):
    df = df.copy()
    vehicles = df["description"].apply(extract_vehicles)
    df[VEHICLE_COLUMNS] = pd.DataFrame(
        [(v1, n1, v2, n2, v3, n3) for (v1, n1), (v2, n2), (v3, n3) in vehicles],
        index=df.index,
    )
    df["injured"] = df["description"].apply(extract_wounded)
    df["deaths"] = df["description"].apply(extract_deaths)

    for col in ["vehicle_1_count", "vehicle_2_count", "vehicle_3_count"]:
        df[col] = df[col].fillna("0").astype(str).apply(convert_to_number)
    df["vehicles_total"] = df["vehicle_1_count"] + df["vehicle_2_count"] + df["vehicle_3_count"]
    return df


def main():
    df = process(pd.read_csv(CLEAN_CSV))
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"{len(df)} rows, {int((df['deaths'] > 0).sum())} fatal -> {PROCESSED_CSV.name}")


if __name__ == "__main__":
    main()
