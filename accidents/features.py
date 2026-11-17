"""Features derived from the processed data.

Only *circumstances* of an accident are used (when, where, who/what was
involved, type of collision). Outcome words (deaths, injuries, people trapped,
hospital) are deliberately excluded: the target labels are extracted from the
same description, so using them as features would leak the answer.
"""
import re

import pandas as pd

from accidents.text import remove_diacritics

# A report describes an accident (and not roadworks, a vehicle fire, an
# oversize transport, a railway incident, ...) when it contains one of these.
REGEX_ACCIDENT = (
    r"accident|coliziun|impact|acrosa|tampona|\blovit|accidentat|rasturn|carambol|"
    r"izbit|derapa\w*|pierd\w* controlul|iesit de pe|paras\w* (?:partial )?(?:partea |suprafata )?carosabil"
)

# Circumstance flags: name -> regex on the lower-case, diacritic-free text
CIRCUMSTANCES = {
    "pedestrian": r"\bpieton",
    "cyclist": r"\bbiciclet|\bbiciclist",
    "motorcycle": r"\bmotocicl|\bmotociclist|\bmoped|\bscuter",
    "frontal": r"\bfrontal",
    "rollover": r"\brasturn",
    "run_off_road": (r"iesit de pe (?:partea )?carosabil|paras\w* (?:partial )?(?:partea |suprafata )?carosabil|"
                     r"in afara (?:partii )?carosabil|afara partii carosabile|\bsant\b|\brapa\b"),
    "fixed_obstacle": r"\bcopac|\barbore|\bstalp|\bparapet|cap de pod|\bpod\b",
    "animal": r"\banimal|\bcaprior|\bmistret|\burs\b|\bcal\b|\bcai\b|\bvac[ai]\b|\bcaine",
    "rail": r"\btren\b|\btrenul|cale ferata|trecere(?:a)? la nivel",
    "heavy_vehicle": r"\btir\b|\btiruri|autocamion|camion|ansambl\w* rutier|autotren|autocisterna|cisterna|basculant",
    "bus": r"\bautobuz|\bmicrobuz|\bautocar",
    "fire": r"\bincendi|\bflacari|\ba luat foc",
    "bad_weather": r"\bpolei|\bzapad|carosabil (?:umed|alunecos)|\bploaie|\bceata",
    "skid": r"\bderapa\w*|pierd\w* controlul",
}


def _text(series):
    return series.fillna("").map(remove_diacritics).str.lower()


def is_accident(descriptions):
    return _text(descriptions).str.contains(REGEX_ACCIDENT)


def circumstance_flags(descriptions):
    text = _text(descriptions)
    return pd.DataFrame(
        {f"has_{name}": text.str.contains(regex).astype(int) for name, regex in CIRCUMSTANCES.items()},
        index=descriptions.index,
    )


def road_type(road):
    """A (motorway), DN (national), DE/DEX (expressway), DNCB (Bucharest ring), other."""
    match = re.match(r"[A-Z]+", road) if isinstance(road, str) else None
    if match is None:
        return "unknown"
    prefix = match.group(0)
    if prefix in ("DEX", "DE"):
        return "DE"
    if prefix == "CENTURA":
        return "DNCB"
    return prefix if prefix in ("A", "DN", "DNCB") else "other"


def time_features(timestamps):
    ts = pd.to_datetime(timestamps)
    return pd.DataFrame({
        "hour": ts.dt.hour,
        "day_of_week": ts.dt.dayofweek,
        "is_weekend": ts.dt.dayofweek.isin([5, 6]).astype(int),
        "month": ts.dt.month,
    }, index=timestamps.index)


CATEGORICAL = ["county", "road_type"]
CONTINUOUS = ["hour", "day_of_week", "month", "vehicles_total"]
BINARY = ["is_weekend"] + [f"has_{name}" for name in CIRCUMSTANCES]
NUMERIC = CONTINUOUS + BINARY


def build_features(df):
    """Feature table (circumstances only) for the rows of `df`."""
    out = pd.concat([
        df[["county"]],
        df["road"].map(road_type).rename("road_type"),
        time_features(df["timestamp"]),
        df[["vehicles_total"]].clip(upper=10),
        circumstance_flags(df["description"]),
    ], axis=1)
    return out[CATEGORICAL + NUMERIC]
