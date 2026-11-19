import numpy as np
import pandas as pd

from accidents.config import COUNTIES
from accidents.geo import load_counties, plotly_county_map


def test_geojson_counties_match_county_list():
    names = {f["properties"]["county"] for f in load_counties()["features"]}
    assert names == set(COUNTIES)


def test_rings_are_clockwise():
    # signed area < 0 for clockwise rings (needed by d3-based renderers)
    for feature in load_counties()["features"]:
        geom = feature["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            x, y = np.asarray(poly[0]).T
            assert 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) < 0


def test_plotly_county_map_handles_missing_counties():
    fig = plotly_county_map(pd.Series({"SIBIU": 3.0, "DOLJ": 8.0}), label="x")
    assert len(fig.data) > 42          # one trace per polygon + colour bar
