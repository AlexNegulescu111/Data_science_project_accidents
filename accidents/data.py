"""Loading and filtering helpers shared by the analysis, models and dashboard."""
import pandas as pd

from accidents.config import (CLEAN_CSV, PROCESSED_CSV, UNKNOWN_COUNTY,
                              VEHICLE_SINGULAR)


def _require(path, step):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found - run `python scripts/{step}.py` "
            f"(or `python run_pipeline.py`) first."
        )


def load_clean():
    _require(CLEAN_CSV, "cleaning_data")
    return pd.read_csv(CLEAN_CSV, parse_dates=["data_ora"])


def load_processed():
    _require(PROCESSED_CSV, "process_data")
    return pd.read_csv(PROCESSED_CSV, parse_dates=["data_ora"])


def known_location(df):
    """Keep only accidents with a known county and road."""
    return df[(df["judet"] != UNKNOWN_COUNTY) & df["judet"].notna() & df["drum"].notna()]


def model_frame(df, columns):
    """Columns needed by a model, rows with a known location and no NaN."""
    cols = list(dict.fromkeys(list(columns) + ["judet", "drum"]))
    return known_location(df)[cols].dropna()


def dangerous_accidents(df, min_wounded=2):
    """Fatal accidents, or accidents with at least `min_wounded` injured."""
    return df[(df["nr_morti"] > 0) | (df["nr_raniti"] >= min_wounded)]


def normalize_vehicles(df, columns=("veh_1", "veh_2", "veh_3")):
    df = df.copy()
    df[list(columns)] = df[list(columns)].replace(VEHICLE_SINGULAR)
    return df
