# Google Earth Engine feature-extraction pipeline

These four scripts regenerate the model input (`data/dados_concatenados.csv`)
from the public **MapBiomas Alerta** polygons. They reproduce the feature
extraction described in the paper: Sentinel-1 SAR backscatter, GLCM texture, and
terrain/hydrology context, per validated alert polygon.

## Prerequisites

1. A Google Earth Engine account and a **cloud project** with the Earth Engine
   API enabled (<https://code.earthengine.google.com/>).
2. `pip install earthengine-api` (already in `requirements.txt`).
3. The MapBiomas alerts uploaded to your project as an asset:
   - Download the alerts **shapefile** from
     <https://plataforma.alerta.mapbiomas.org/downloads>.
   - In the Earth Engine Code Editor → *Assets* → *New* → *Table upload*, upload
     the shapefile. Note the resulting asset path.
   - The asset must expose the properties used by the scripts:
     `CODEALERTA`, `ESTADO`, `BIOMA`, `VPRESSAO`, `DTIMGDEP` (detection date, ms).

## Configuration (environment variables)

```bash
export GEE_PROJECT=ee-your-project
export GEE_ASSET=projects/ee-your-project/assets/<your-alerts-asset>
```

Both are read via `config.py`. No project id or asset path is hard-coded.

## Run order

```bash
# 1) Sample + validate Sentinel-1 coverage → writes a GEE asset
#    projects/${GEE_PROJECT}/assets/master_sample_validated
python gee/01_explore_and_sample.py

# 2) SAR + GLCM features  → CSVs to Google Drive folder "SAR_features"
python gee/02_extract_sar_glcm.py

# 3) terrain / river context → CSVs to the same Drive folder
python gee/03_extract_context.py

# --- wait for the GEE tasks to finish (https://code.earthengine.google.com/tasks),
#     then download the per-state CSVs from Drive into one local folder ---

# 4) merge SAR+GLCM with context, concatenate the four states
python gee/04_merge_features.py --input-dir /path/to/downloaded/csvs
# → writes data/dados_concatenados.csv
```

## Pipeline diagram

```
MapBiomas Alerta shapefile
        │  (download + upload to GEE)
        ▼
GEE_ASSET  ──01──►  master_sample_validated (asset, S1-covered polygons)
                        │
              ┌─────────┴─────────┐
             02                  03
        SAR + GLCM CSVs     context CSVs      (per state → Google Drive)
              └─────────┬─────────┘
                       04  (download + merge)
                        ▼
             data/dados_concatenados.csv  →  src/sase_spatial.py
```

## Method notes (as in the paper)

- **Scene selection**: first Sentinel-1 GRD IW, VV+VH, DESCENDING scene within
  30 days of each alert's detection date (`DTIMGDEP`).
- **Speckle filtering**: Refined Lee, applied in linear power domain (`02_...`).
- **GLCM**: `glcmTexture(size=7)` on int16-scaled VV/VH; metrics contrast,
  dissimilarity, IDM, ASM, correlation, entropy.
- **Region reduction**: mean, stdDev, and 10th/90th percentiles over each polygon
  (scale 10 m for SAR/GLCM, 30 m for terrain context).
- **Context**: distance to nearest free-flowing river (HydroSHEDS), flow order,
  slope, roughness (elevation std-dev), and TPI, from NASADEM.
