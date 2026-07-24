"""
GEE step 1 — Explore the MapBiomas alerts, build the sampling plan, validate
Sentinel-1 coverage, and export a validated master sample as a GEE asset.

Pipeline position
-----------------
    [MapBiomas alerts shapefile]  →  upload to GEE as an asset (see gee/README.md)
    01_explore_and_sample.py   ← YOU ARE HERE   → asset: <project>/assets/master_sample_validated
    02_extract_sar_glcm.py     → per-state SAR + GLCM CSVs (to Drive)
    03_extract_context.py      → per-state terrain/river context CSVs (to Drive)
    04_merge_features.py       → dados_concatenados.csv (model input)

What it does
------------
  * Restrict alerts to the Amazônia biome and to the four study states.
  * Keep all `ilegal_mining` polygons; random-sample the "resto" (non-mining)
    classes down to the per-state cap (reproducible via seed=2025).
  * Check that a Sentinel-1 IW/VV+VH DESCENDING scene exists within 30 days of
    each alert's detection date (DTIMGDEP); drop polygons without coverage.
  * Export the validated sample as a GEE FeatureCollection asset.

Configuration (environment variables — see config.py / gee/README.md)
--------------------------------------------------------------------
  GEE_PROJECT   your Earth Engine cloud project id (e.g. "ee-your-project")
  GEE_ASSET     asset path of the uploaded MapBiomas alerts FeatureCollection,
                e.g. "projects/${GEE_PROJECT}/assets/mapbiomas-alerts"

Usage
-----
  export GEE_PROJECT=ee-your-project
  export GEE_ASSET=projects/ee-your-project/assets/mapbiomas-alerts
  python gee/01_explore_and_sample.py
"""

import sys
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import require_gee_project, GEE_ASSET

PROJECT = require_gee_project()
if not GEE_ASSET:
    raise SystemExit(
        "GEE_ASSET is not set. Upload the MapBiomas alerts shapefile to your GEE\n"
        "project and export its asset path, e.g.:\n"
        f"    export GEE_ASSET=projects/{PROJECT}/assets/mapbiomas-alerts\n"
        "See gee/README.md for the download + upload steps."
    )

# Output asset for the validated master sample (created by this script).
VALIDATED_ASSET = f"projects/{PROJECT}/assets/master_sample_validated"

# Per-state sampling plan: keep all illegal-mining; cap the "resto" classes.
SAMPLING_PLAN = {
    'AMAZONAS':    {'ilegal_mining': 'ALL', 'resto': 4000},
    'MATO GROSSO': {'ilegal_mining': 'ALL', 'resto': 4000},
    'PARÁ':        {'ilegal_mining': 'ALL', 'resto': 6759},
    'RORAIMA':     {'ilegal_mining': 'ALL', 'resto': 4000},
}
RESTO_CLASSES = ['agriculture', 'aquaculture', 'natural_cause', 'roads', 'urban_expansion']
STATES = list(SAMPLING_PLAN.keys())
SAMPLE_SEED = 2025


def main():
    ee.Authenticate()
    ee.Initialize(project=PROJECT)
    print(f"--- Step 1: sampling, Sentinel-1 validation, asset export ---")
    print(f"    Project: {PROJECT}")
    print(f"    Alerts asset: {GEE_ASSET}")

    dashboard = ee.FeatureCollection(GEE_ASSET)
    amazonia = (dashboard
                .filter(ee.Filter.eq('BIOMA', 'Amazônia'))
                .filter(ee.Filter.notNull(['DTIMGDEP'])))

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))

    def check_s1_coverage(feature):
        geom = feature.geometry()
        start = ee.Date(feature.get('DTIMGDEP'))
        end = start.advance(30, 'day')
        n = s1.filterBounds(geom).filterDate(start, end).size()
        return feature.set('has_s1_image', ee.Algorithms.If(n.gt(0), 1, 0))

    # --- Sampling (server-side, held in memory) ---
    print("... sampling per state ...")
    collections = []
    for state, plan in SAMPLING_PLAN.items():
        print(f"    Sampling: {state}")
        subset = amazonia.filter(ee.Filter.eq('ESTADO', state))
        ilegal = subset.filter(ee.Filter.eq('VPRESSAO', 'ilegal_mining'))

        resto_orig = subset.filter(ee.Filter.inList('VPRESSAO', RESTO_CLASSES))
        n_resto = resto_orig.size().getInfo()
        cap = plan['resto']
        if cap == 'ALL' or cap >= n_resto:
            resto_sel = resto_orig
        else:
            resto_sel = resto_orig.randomColumn(seed=SAMPLE_SEED).sort('random').limit(cap)
        resto_sel = resto_sel.map(lambda f: f.set('VPRESSAO', 'resto'))

        collections.append(ilegal)
        collections.append(resto_sel)

    raw_sample = ee.FeatureCollection(collections).flatten()
    print(f"    Raw sampled features: {raw_sample.size().getInfo()}")

    # --- Sentinel-1 coverage check + report ---
    print("... checking Sentinel-1 coverage ...")
    checked = raw_sample.map(check_s1_coverage)

    rows = []
    for state in STATES:
        for cls in ['ilegal_mining', 'resto']:
            sub = (checked.filter(ee.Filter.eq('ESTADO', state))
                          .filter(ee.Filter.eq('VPRESSAO', cls)))
            n = sub.size().getInfo()
            n_ok = sub.filter(ee.Filter.eq('has_s1_image', 1)).size().getInfo() if n else 0
            rows.append({'state': state, 'class': cls, 'total': n,
                         'with_s1': n_ok, 'without_s1': n - n_ok,
                         'coverage_%': round(100 * n_ok / n, 2) if n else 0})
    print("\n=== Sentinel-1 coverage report (raw sample) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # --- Keep only validated (S1-covered) polygons and export asset ---
    validated = checked.filter(ee.Filter.eq('has_s1_image', 1))
    print(f"\n    Validated features (with S1): {validated.size().getInfo()}")

    task = ee.batch.Export.table.toAsset(
        collection=validated,
        description="export_master_sample_validated",
        assetId=VALIDATED_ASSET,
    )
    task.start()
    print(f"\n✅ Export task started. Asset: {VALIDATED_ASSET}")
    print("   Monitor progress at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
