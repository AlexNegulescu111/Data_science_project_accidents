"""Sanity checks on the generated CSVs (run after the pipeline)."""
import pandas as pd
import pytest

from accidents.config import CLEAN_CSV, COUNTIES, PROCESSED_CSV, UNKNOWN_COUNTY

pytestmark = pytest.mark.skipif(not PROCESSED_CSV.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def clean():
    return pd.read_csv(CLEAN_CSV)


@pytest.fixture(scope="module")
def processed():
    return pd.read_csv(PROCESSED_CSV)


def test_counties_are_valid(clean):
    assert set(clean["county"]) <= set(COUNTIES) | {UNKNOWN_COUNTY}


def test_no_diacritics_left(clean):
    assert not clean["description"].str.contains("[ăâîșțşţĂÂÎȘȚŞŢ]").any()


def test_resumption_is_plausible(clean):
    resumption = clean["resumption_min"].dropna()
    assert (resumption >= 0).all()
    assert resumption.max() <= 12 * 60


def test_every_death_mention_is_counted(processed):
    mentions = processed["description"].str.lower().str.contains(r"\bdeces|decedat|murit|si-a pierdut viata")
    assert (processed.loc[mentions, "deaths"] > 0).all()


def test_no_index_column(processed):
    assert not any(c.startswith("Unnamed") for c in processed.columns)
