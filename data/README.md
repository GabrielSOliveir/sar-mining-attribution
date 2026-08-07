# Data

## `dados_concatenados.csv` — the model input (not committed; ~135 MB)

This is the single, self-contained input to the SASE model and every analysis /
figure script. One row per validated alert polygon, with columns:

- `CODEALERTA` — alert id (join key / grouping id)
- `ESTADO` — state (`PARÁ`, `MATO GROSSO`, `AMAZONAS`, `RORAIMA`)
- `VPRESSAO` — binarized label: `ilegal_mining` (positive) or `resto` (negative)
- Sentinel-1 backscatter: `VV_*`, `VH_*`, `VV_minus_VH_*` (`mean`/`stdDev`/`p10`/`p90`)
- GLCM texture: `contrast_*`, `diss_*`, `corr_*`, `asm_*`, `ent_*`, `idm_*` for `vv`/`vh`
- Terrain / hydrology context: `dist_river`, `slope`, `roughness`, `tpi`
- `.geo` — polygon geometry as GeoJSON (WGS84), used for spatial blocking and the map

The raw CSV is **not committed** (GitHub rejects files > 100 MB). Instead, the
dataset ships **zipped** in this directory as `dados_concatenados.csv.zip`
(~49 MB) — the scripts read it directly (pandas decompresses on the fly, see
`config.py`), so no manual step is needed after cloning. Obtain / restore it any
of these ways:

### Option A — use the bundled zip (default; nothing to do)
`data/dados_concatenados.csv.zip` is included in the repository. All scripts pick
it up automatically. To materialize the raw CSV (slightly faster reads):
`unzip data/dados_concatenados.csv.zip -d data/`.

### Option A′ — download from this repository
The same zipped file can be downloaded directly from the repository page
(<https://github.com/GabrielSOliveir/sar-mining-attribution/blob/main/data/dados_concatenados.csv.zip>)
without cloning.

### Option B — regenerate it from scratch on Google Earth Engine
Run the four scripts in [`../gee/`](../gee) — see [`../gee/README.md`](../gee/README.md).
The source alerts come from the **MapBiomas Alerta** platform:

1. Download the alerts **shapefile** from
   <https://plataforma.alerta.mapbiomas.org/downloads>.
2. Upload it to your Google Earth Engine project as a table asset.
3. Run `gee/01_explore_and_sample.py` → `02_extract_sar_glcm.py` →
   `03_extract_context.py`, download the exported CSVs, then run
   `gee/04_merge_features.py` to produce `dados_concatenados.csv`.

> Note: exact backscatter values depend on the Sentinel-1 GRD scenes available at
> query time; regenerated features should match the published CSV closely but may
> differ at the last decimals. For bit-exact reproduction of the paper's numbers,
> use Option A.

## Small reference tables (committed)

These are small derived tables kept in the repo for convenience:

- `feature_separability.csv` — per-state Cliff's Delta separability (also produced
  by `src/feature_analysis.py`).
- `morphometry_by_state.csv`, `resto_composition_by_state.csv`,
  `vpressao_por_estado_amazonia.csv` — class-composition / polygon-size summaries.
