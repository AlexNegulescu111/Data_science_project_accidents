"""Matplotlib setup: figures are saved to `visuals/`, and only shown on screen
when SHOW_PLOTS=1 (or `--show` is passed to run_pipeline.py)."""
import os

import matplotlib

SHOW = os.environ.get("SHOW_PLOTS", "0") == "1"
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from accidents.config import VISUALS_DIR  # noqa: E402


def save_figure(name, fig=None, subdir=None):
    """Save the current (or given) figure as visuals/[subdir/]name.png."""
    fig = fig or plt.gcf()
    out_dir = VISUALS_DIR / subdir if subdir else VISUALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path.relative_to(VISUALS_DIR.parent)}")
    if SHOW:
        plt.show()
    plt.close(fig)
    return path
