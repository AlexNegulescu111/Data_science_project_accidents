"""County boundaries (simplified GeoJSON, source: click_that_hood by Code for
America, MIT License) and a matplotlib choropleth without extra dependencies."""
import json
from functools import lru_cache

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from accidents.config import GEO_COUNTIES


@lru_cache(maxsize=1)
def load_counties():
    """GeoJSON dict; every feature has id/properties.county = upper-case name."""
    with open(GEO_COUNTIES, encoding="utf-8") as f:
        return json.load(f)


def _polygons(geometry):
    coords = geometry["coordinates"]
    rings = [coords] if geometry["type"] == "Polygon" else coords
    return [np.asarray(poly[0]) for poly in rings]   # outer rings only


def choropleth(ax, values, cmap="Reds", label="", fmt="{:.0f}", annotate=True):
    """Draw counties coloured by `values` (Series indexed by county name)."""
    import matplotlib as mpl

    patches, colours, centres = [], [], {}
    vmin, vmax = np.nanmin(values.values), np.nanmax(values.values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = mpl.colormaps[cmap]
    for feature in load_counties()["features"]:
        county = feature["properties"]["county"]
        value = values.get(county, np.nan)
        for ring in _polygons(feature["geometry"]):
            patches.append(Polygon(ring, closed=True))
            colours.append("#e5e5e5" if np.isnan(value) else colormap(norm(value)))
        biggest = max(_polygons(feature["geometry"]), key=len)
        centres[county] = biggest.mean(axis=0)
    ax.add_collection(PatchCollection(patches, facecolor=colours, edgecolor="white", linewidth=0.6))
    if annotate:
        for county, (x, y) in centres.items():
            if county in values.index and not np.isnan(values[county]):
                ax.text(x, y, fmt.format(values[county]), ha="center", va="center", fontsize=7)
    ax.autoscale_view()
    ax.set_aspect(1.4)          # roughly correct at Romania's latitude
    ax.axis("off")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    ax.figure.colorbar(sm, ax=ax, shrink=0.7, label=label)


def plotly_county_map(values, hover=None, colorscale="Reds", label="", height=520):
    """Interactive county map drawn as filled polygons (plotly).

    Unlike plotly's geo/map traces it needs no base-map download and no WebGL,
    so it renders offline and in any browser.
    `values`: Series indexed by county; `hover`: optional dict county -> text.
    """
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale

    vmin, vmax = float(np.nanmin(values.values)), float(np.nanmax(values.values))
    span = (vmax - vmin) or 1.0
    fig = go.Figure()
    for feature in load_counties()["features"]:
        county = feature["properties"]["county"]
        value = values.get(county, np.nan)
        colour = "#e5e5e5" if np.isnan(value) else sample_colorscale(colorscale, (value - vmin) / span)[0]
        text = (hover or {}).get(county, f"{county}: {value:g}" if not np.isnan(value) else f"{county}: no data")
        for ring in _polygons(feature["geometry"]):
            fig.add_trace(go.Scatter(
                x=ring[:, 0], y=ring[:, 1], fill="toself", fillcolor=colour, mode="lines",
                line=dict(color="white", width=0.8), hoverinfo="text", text=text,
                hoveron="fills", showlegend=False,
            ))
    # invisible trace that only carries the colour bar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=colorscale, cmin=vmin, cmax=vmax, color=[vmin],
                    colorbar=dict(title=label, thickness=14)),
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1.4)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
