"""
Central configuration and path resolution for the reproducibility repository.

All scripts import from here instead of hard-coding absolute paths, so the
repository is portable across machines. Paths are resolved in this order:

    1. Environment variable (if set)
    2. A sensible default relative to the repository root

Environment variables
----------------------
DATA_DIR      Directory holding the input CSV(s). Default: <repo>/data
RESULTS_DIR   Directory where model outputs are written. Default: <repo>/results
FIGURES_DIR   Directory where figures are written. Default: <repo>/figures
GEE_PROJECT   Google Earth Engine cloud project id (e.g. "ee-your-project").
              Required only for the scripts under gee/. No default.
GEE_ASSET     Full asset path of the MapBiomas alerts FeatureCollection uploaded
              to your GEE project (see gee/README.md). Required only for gee/.

Example
-------
    export DATA_DIR=/path/to/data
    export GEE_PROJECT=ee-your-project
    python src/sase_spatial.py
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _resolve(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else default


# ── Data / output directories ────────────────────────────────────────────────
DATA_DIR    = _resolve("DATA_DIR",    REPO_ROOT / "data")
RESULTS_DIR = _resolve("RESULTS_DIR", REPO_ROOT / "results")
FIGURES_DIR = _resolve("FIGURES_DIR", REPO_ROOT / "figures")

# ── Canonical input file (model input; self-contained) ────────────────────────
# SAR + GLCM + terrain/river context features per validated alert polygon.
# The repository ships the data as a zip (dados_concatenados.csv.zip, ~49 MB);
# pandas.read_csv() reads it directly. If you unzip it, the raw .csv is preferred
# automatically (slightly faster). Regenerate from MapBiomas alerts with the
# scripts under gee/ (see gee/README.md).
_csv = DATA_DIR / "dados_concatenados.csv"
_zip = DATA_DIR / "dados_concatenados.csv.zip"
DATA_CSV = _csv if _csv.exists() else _zip

# ── Google Earth Engine (only needed to regenerate features from scratch) ─────
GEE_PROJECT = os.environ.get("GEE_PROJECT", "")
# Full asset path of the MapBiomas alerts shapefile uploaded to your GEE project.
# Example: f"projects/{GEE_PROJECT}/assets/mapbiomas-alerts"
GEE_ASSET = os.environ.get("GEE_ASSET", "")


def require_gee_project() -> str:
    """Return GEE_PROJECT or raise a helpful error if unset."""
    if not GEE_PROJECT:
        raise SystemExit(
            "GEE_PROJECT is not set. Export your Earth Engine project id first:\n"
            "    export GEE_PROJECT=ee-your-project\n"
            "See gee/README.md for setup instructions."
        )
    return GEE_PROJECT
