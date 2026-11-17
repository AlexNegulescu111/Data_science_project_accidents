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
    _require(CLEAN_CSV, "clean_data")
    return pd.read_csv(CLEAN_CSV, parse_dates=["timestamp"])


def load_processed():
    _require(PROCESSED_CSV, "process_data")
    return pd.read_csv(PROCESSED_CSV, parse_dates=["timestamp"])


def known_location(df):
    """Keep only accidents with a known county and road."""
    return df[(df["county"] != UNKNOWN_COUNTY) & df["county"].notna() & df["road"].notna()]


def model_frame(df, columns):
    """Columns needed by a model, rows with a known location and no NaN."""
    cols = list(dict.fromkeys(list(columns) + ["county", "road"]))
    return known_location(df)[cols].dropna()


def dangerous_accidents(df, min_wounded=2):
    """Fatal accidents, or accidents with at least `min_wounded` injured."""
    return df[(df["deaths"] > 0) | (df["injured"] >= min_wounded)]


def normalize_vehicles(df, columns=("vehicle_1", "vehicle_2", "vehicle_3")):
    df = df.copy()
    df[list(columns)] = df[list(columns)].replace(VEHICLE_SINGULAR)
    return df
