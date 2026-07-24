"""
OPTIONAL data-prep — attach the ORIGINAL (multi-class) MapBiomas VPRESSAO label
to each alert, alongside the binarized label used for modeling.

The model input (dados_concatenados.csv) carries VPRESSAO already binarized into
'ilegal_mining' / 'resto'. This script queries the source MapBiomas alerts asset
on GEE for the original class of each CODEALERTA (agriculture, roads, mining,
aquaculture, …) and writes dados_completos.csv = all original columns +
VPRESSAO_original. Used only for the class-composition tables/figures; not
required to reproduce the SASE results.

Requires GEE. Configure via environment (see config.py / gee/README.md):
  GEE_PROJECT, GEE_ASSET

Usage
-----
  export GEE_PROJECT=ee-your-project
  export GEE_ASSET=projects/ee-your-project/assets/mapbiomas-alerts
  python src/data_prep/build_dados_completos.py
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

CSV_IN  = str(DATA_CSV)
CSV_OUT = str(DATA_DIR / "dados_completos.csv")
BATCH   = 4000

ee.Initialize(project=PROJECT)

print("1) Reading CSV ...")
df = pd.read_csv(CSV_IN)
print(f"   CSV: {df.shape[0]} rows, {df.shape[1]} cols")
codes = sorted(set(int(c) for c in df['CODEALERTA'].dropna().unique()))
print(f"   unique CODEALERTA: {len(codes)}")

print("2) Querying GEE in batches ...")
fc = ee.FeatureCollection(GEE_ASSET)
mapping = {}
for i in range(0, len(codes), BATCH):
    lote = codes[i:i + BATCH]
    sub = (fc.filter(ee.Filter.inList('CODEALERTA', lote))
             .select(['CODEALERTA', 'VPRESSAO'], None, False))
    feats = sub.getInfo()['features']
    for f in feats:
        p = f['properties']
        mapping[int(p['CODEALERTA'])] = p['VPRESSAO']
    print(f"   batch {i//BATCH + 1}: +{len(feats)} features (cumulative {len(mapping)})")

print("3) Joining ...")
df['VPRESSAO_original'] = df['CODEALERTA'].astype(int).map(mapping)
n_ok = df['VPRESSAO_original'].notna().sum()
print(f"   mapped: {n_ok}/{len(df)} (missing {len(df) - n_ok})")
print("\n   VPRESSAO_original distribution:")
print(df['VPRESSAO_original'].value_counts(dropna=False).to_string())

df.to_csv(CSV_OUT, index=False)
print(f"\n✅ Saved: {CSV_OUT}  ({df.shape[0]} rows, {df.shape[1]} cols)")
