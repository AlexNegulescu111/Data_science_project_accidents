"""Step 2: raw scraped CSV -> data/accidents_clean.csv

Removes diacritics and bad rows, extracts county and road from the title,
computes the traffic resumption time in minutes and builds a timestamp.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from accidents.config import CLEAN_CSV, MONTHS_RO_EN, RAW_CSV, UNKNOWN_COUNTY
from accidents.text import (extract_county, extract_road, remove_diacritics,
                            resumption_minutes)


def clean(df):
    df = df.map(remove_diacritics)

    # Rows without time/description, or whose description does not start the
    # standard way ("Centrul INFOTRAFIC"), are bad scrapes.
    df = df.dropna(subset=["Ora", "Descriere"])
    df = df[df["Descriere"].str.contains("Centrul INFOTRAFIC")]
    df = df.drop_duplicates(subset="Descriere", keep="first")

    out = pd.DataFrame(index=df.index)
    out["county"] = df["Titlu"].map(extract_county).fillna(UNKNOWN_COUNTY)
    out["road"] = df["Titlu"].map(extract_road)

    report_time = df["Ora"].str.replace("Ora: ", "", regex=False).str.strip()
    out["resumption_min"] = pd.array(
        [resumption_minutes(o, r, d) for o, r, d in zip(report_time, df["Reluare"], df["Descriere"])],
        dtype="Int64",
    )

    # "Data: 03 Iulie 2025" + "15:50" -> timestamp
    months_regex = r"\b(" + "|".join(MONTHS_RO_EN) + r")\b"
    data_en = (df["Data"].str.replace("Data: ", "", regex=False)
               .str.replace(months_regex, lambda m: MONTHS_RO_EN[m.group(0)], regex=True))
    date = pd.to_datetime(data_en, format="%d %B %Y", errors="coerce")
    out["timestamp"] = pd.to_datetime(
        date.dt.strftime("%Y-%m-%d") + " " + report_time, format="%Y-%m-%d %H:%M", errors="coerce"
    )
    out["description"] = df["Descriere"]
    return out[["timestamp", "county", "road", "description", "resumption_min"]]


def main():
    raw = pd.read_csv(RAW_CSV)
    clean_df = clean(raw)
    clean_df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"{len(raw)} raw rows -> {len(clean_df)} clean rows -> {CLEAN_CSV.name}")


if __name__ == "__main__":
    main()
