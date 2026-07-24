"""
OPTIONAL data-prep — fetch geodesic geometry (area, perimeter) per alert polygon
from the source MapBiomas alerts asset on GEE.

Writes a cache CSV (CODEALERTA, area_m2, perimeter_m, n_match). Used only for the
polygon-morphometry table in the paper; not required to reproduce SASE results.

Requires GEE. Configure via environment (see config.py / gee/README.md):
  GEE_PROJECT, GEE_ASSET

Usage
-----
  export GEE_PROJECT=ee-your-project
  export GEE_ASSET=projects/ee-your-project/assets/mapbiomas-alerts
  python src/data_prep/fetch_morphometry.py
"""

import sys
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_DIR, DATA_CSV, require_gee_project, GEE_ASSET

PROJECT = require_gee_project()
if not GEE_ASSET:
    raise SystemExit("GEE_ASSET is not set (see gee/README.md).")

CSV_IN = str(DATA_CSV)
CACHE  = str(DATA_DIR / "_morphometry_raw.csv")
BATCH  = 4000

ee.Initialize(project=PROJECT)
df = pd.read_csv(CSV_IN)
codes = sorted(set(int(c) for c in df['CODEALERTA'].dropna().unique()))
print(f"unique CODEALERTA: {len(codes)}")

fc = ee.FeatureCollection(GEE_ASSET)


def add_morph(f):
    g = f.geometry()
    return ee.Feature(None, {
        'CODEALERTA': f.get('CODEALERTA'),
        'area_m2': g.area(1),          # maxError=1, geodesic, m^2
        'perimeter_m': g.perimeter(1)  # maxError=1, geodesic, m
    })


recs = {}
for i in range(0, len(codes), BATCH):
    lote = codes[i:i + BATCH]
    sub = fc.filter(ee.Filter.inList('CODEALERTA', lote)).map(add_morph)
    feats = sub.getInfo()['features']
    for f in feats:
        p = f['properties']
        c = int(p['CODEALERTA'])
        if c in recs:
            recs[c]['n_match'] += 1
        else:
            recs[c] = {'CODEALERTA': c, 'area_m2': p['area_m2'],
                       'perimeter_m': p['perimeter_m'], 'n_match': 1}
    print(f"   batch {i//BATCH+1}: {len(feats)} features (distinct CODEALERTA {len(recs)})")

morph = pd.DataFrame(recs.values())
morph.to_csv(CACHE, index=False)
n_dup = (morph['n_match'] > 1).sum()
print(f"\nCODEALERTA with geometry: {len(morph)} | missing: {len(codes) - len(morph)} | >1 feature: {n_dup}")
print(f"✅ cache saved: {CACHE}")
