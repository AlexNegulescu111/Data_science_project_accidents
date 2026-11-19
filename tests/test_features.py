import re

import numpy as np
import pandas as pd
import pytest

from accidents.features import (CIRCUMSTANCES, NUMERIC, build_features,
                                circumstance_flags, is_accident, road_type)
from accidents.stats import shrunk_rate, wilson_ci
from accidents.text import REGEX_DEATHS


@pytest.mark.parametrize("text, expected", [
    ("in urma unei coliziuni in care au fost implicate un autoturism si un autotren", True),
    ("pentru a permite interventia la locul producerii unui accident rutier", True),
    ("conducatorul unui ansamblu TIR pierde controlul directiei, paraseste suprafata carosabila", True),
    ("un autoturism a parasit partial carosabilul pe un sector de drum in curba", True),
    ("un pieton a fost accidentat mortal", True),
    # not accidents
    ("s-a produs un incendiu asupra unui autoturism, fara a rezulta victime", False),
    ("se vor desfasura 2 transporturi agabaritice pe tronsonul DN 11", False),
    ("au loc lucrari de montare a unui portal", False),
])
def test_is_accident(text, expected):
    assert is_accident(pd.Series([text])).iloc[0] == expected


def test_circumstance_flags():
    text = pd.Series([
        "Un autoturism a acrosat un pieton.",
        "Coliziune frontala intre un autoturism si un TIR.",
        "Un autoturism a intrat intr-un copac, in afara localitatii.",
        "Un autoturism a lovit un cal pe DN 1.",
    ])
    flags = circumstance_flags(text)
    assert flags.loc[0, "has_pedestrian"] == 1
    assert flags.loc[1, ["has_frontal", "has_heavy_vehicle"]].tolist() == [1, 1]
    assert flags.loc[2, "has_fixed_obstacle"] == 1
    assert flags.loc[3, "has_animal"] == 1
    # "localitatii" must not trigger the horse ("cal") pattern
    assert flags.loc[2, "has_animal"] == 0


@pytest.mark.parametrize("pattern", CIRCUMSTANCES.values())
def test_circumstances_do_not_encode_the_outcome(pattern):
    # Features must describe circumstances, never deaths (that is the label)
    for word in ["decedat", "decesul", "si-a pierdut viata", "mortal", "a murit"]:
        assert not re.search(pattern, word)
    assert not re.search(REGEX_DEATHS, pattern)


@pytest.mark.parametrize("road, expected", [
    ("DN 7", "DN"), ("A 1", "A"), ("DNCB", "DNCB"), ("CENTURA BUCURESTI", "DNCB"),
    ("DEX 12", "DE"), ("DJ 102", "other"), (None, "unknown"), ("unknown", "unknown"),
])
def test_road_type(road, expected):
    assert road_type(road) == expected


def test_build_features_has_no_outcome_columns_and_no_nan():
    df = pd.DataFrame({
        "county": ["SIBIU", "UNKNOWN"],
        "road": ["A 1", None],
        "timestamp": pd.to_datetime(["2024-05-04 15:50", "2024-05-06 08:10"]),
        "vehicles_total": [2, 15],
        "description": ["coliziune frontala", "un pieton a fost lovit"],
        "deaths": [1, 0], "injured": [2, 0], "resumption_min": [30, 60],
    })
    X = build_features(df)
    assert not {"deaths", "injured", "resumption_min"} & set(X.columns)
    assert X.notna().all().all()
    assert X["is_weekend"].tolist() == [1, 0]
    assert X["vehicles_total"].max() == 10          # clipped
    assert set(NUMERIC) <= set(X.columns)


def test_wilson_ci_known_values():
    low, high = wilson_ci(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0, abs=1e-12) and 0 < high < 0.35


def test_wilson_ci_vectorised_and_narrows_with_n():
    low, high = wilson_ci(np.array([2, 20]), np.array([10, 100]))
    assert (high - low)[1] < (high - low)[0]


def test_shrunk_rate_pulls_small_groups_to_prior():
    # 1 of 2 fatal is not "50 % risk"
    assert shrunk_rate(1, 2, prior_rate=0.1) < 0.15
    # with many observations the observed rate dominates
    assert shrunk_rate(500, 1000, prior_rate=0.1) == pytest.approx(0.49, abs=0.01)
