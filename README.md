# Regional Specificity of SAR-Based Illegal Mining Signatures in the Brazilian Amazon

Reproducibility code and Google Earth Engine scripts for the paper submitted to
*Remote Sensing* (MDPI).

> **Double-blind review.** Author names, affiliations, project ids, and account
> identifiers have been removed or parameterized. Nothing in this repository
> should reveal author identity; please open an issue via the editorial system if
> you find a leak.

## What this does

We attribute the driver of each validated deforestation alert — **illegal mining
(`ilegal_mining`) vs. everything else (`resto`)** — from **Sentinel-1 C-band SAR**
features (VV, VH, VV−VH backscatter; GLCM texture; terrain/hydrology context),
using a per-state **State-Aware Stacking Ensemble (SASE)**. We evaluate across
four Amazonian states — **Pará, Mato Grosso, Amazonas, Roraima** — with a
spatially-blocked split, and test cross-state generalization (Pará → others). The
central finding: SAR mining signatures are **regionally specific**, so a model
trained on one state does not transfer well to the others.

## Headline results (reproduced by `src/sase_spatial.py`)

Aggregate test-set metrics over 5 spatial splits (seeds 42–46), spatial block
resolution 0.2° (~22 km). These match
[`results/reference_res020/sase_summary.csv`](results/reference_res020/sase_summary.csv).

| State        | F1              | AUC-ROC         | Precision (mining) | Recall (mining) |
|--------------|-----------------|-----------------|--------------------|-----------------|
| Pará         | **0.881 ± 0.006** | 0.943 ± 0.004 | 0.899              | 0.864           |
| Mato Grosso  | **0.742 ± 0.080** | 0.956 ± 0.014 | 0.674              | 0.840           |
| Amazonas     | **0.520 ± 0.089** | 0.866 ± 0.030 | 0.517              | 0.538           |
| Roraima      | **0.478 ± 0.091** | 0.904 ± 0.021 | 0.411              | 0.572           |

## Repository layout

```
.
├── config.py                 # portable path / GEE config (env-var driven; no hard-coded paths)
├── requirements.txt          # pip dependencies (Python 3.12)
├── environment.yml           # conda alternative (bundles libomp for XGBoost)
├── data/
│   ├── README.md             # how to obtain / regenerate the input dataset
│   └── *.csv                 # small reference tables (large input is gitignored)
├── gee/                      # Google Earth Engine feature-extraction pipeline
│   ├── README.md
│   ├── 01_explore_and_sample.py
│   ├── 02_extract_sar_glcm.py
│   ├── 03_extract_context.py
│   └── 04_merge_features.py
├── src/
│   ├── sase_spatial.py       # MAIN: SASE ensemble → reproduces the headline table
│   ├── feature_analysis.py   # Cliff's Delta separability + distribution figures
│   └── data_prep/            # OPTIONAL GEE-based data-prep (class labels, morphometry)
├── figures/
│   └── figure2_map.py        # Figure 2: study-area map of validated alerts
├── results/
│   └── reference_res020/     # committed reference outputs (the paper's numbers)
└── docs/
    └── experiment_log.md      # narrative log of all experiments behind the paper
```

## Quickstart

```bash
# 1) Environment (Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: XGBoost needs the OpenMP runtime
#   brew install libomp
# (or use conda:  conda env create -f environment.yml && conda activate sar-mining)

# 2) Get the input data (~135 MB) — see data/README.md
#    Option A: download data/dados_concatenados.csv from the paper's data archive
#    Option B: regenerate it via the gee/ pipeline (needs an Earth Engine account)

# 3) Reproduce the headline results + per-state figures
python src/sase_spatial.py
#    → results/sase_res020/  (sase_summary.csv, per-state reports/curves,
#                             generalization_summary.csv)

# 4) Reproduce the supporting analyses / figures
python src/feature_analysis.py     # → results/feature_analysis/
python figures/figure2_map.py      # → figures/figura2_mapa.{pdf,png}
```

Compare `results/sase_res020/sase_summary.csv` against
`results/reference_res020/sase_summary.csv` — the F1 values should match.

## Configuration

All paths are resolved in [`config.py`](config.py) from environment variables,
with sensible repo-relative defaults — **no machine-specific absolute paths**:

| Variable      | Purpose                                   | Default        |
|---------------|-------------------------------------------|----------------|
| `DATA_DIR`    | location of `dados_concatenados.csv`      | `./data`       |
| `RESULTS_DIR` | where model outputs are written           | `./results`    |
| `FIGURES_DIR` | where figures are written                 | `./figures`    |
| `GEE_PROJECT` | Earth Engine cloud project id (`gee/` only) | *(unset)*    |
| `GEE_ASSET`   | uploaded MapBiomas alerts asset path (`gee/` only) | *(unset)* |

```bash
export DATA_DIR=/path/to/data
python src/sase_spatial.py --block-res 0.2 --n-splits 5 --seed 42
```

## Method summary

- **Features** (per validated alert polygon): Sentinel-1 VV/VH/VV−VH backscatter
  (mean, stdDev, p10, p90), GLCM texture (contrast, dissimilarity, correlation,
  ASM, entropy, IDM), and terrain/hydrology context (distance to river, slope,
  roughness, TPI). Feature sets are selected **per state** (see `STATE_FEATURES`).
- **Model (SASE)**: base learners Random Forest + XGBoost + Logistic Regression +
  MLP; meta-learner Logistic Regression, trained on **out-of-fold** predictions
  with folds **grouped by spatial block** (leak-free stacking).
- **Spatial validation**: `GroupShuffleSplit` over 0.2° grid blocks, 70/15/15
  train/val/test; decision threshold τ* optimized on validation only; metrics
  reported on the untouched test set; mean ± std over 5 splits (seeds 42–46).
- **Data**: MapBiomas Alerta deforestation alerts
  (<https://plataforma.alerta.mapbiomas.org/downloads>); Sentinel-1 GRD; NASADEM;
  HydroSHEDS free-flowing rivers.

See [`docs/experiment_log.md`](docs/experiment_log.md) for the full experiment
narrative and [`gee/README.md`](gee/README.md) for the data-generation pipeline.

## License

Code released under the [MIT License](LICENSE) (copyright withheld for
double-blind review). Input alert data is © MapBiomas, distributed under its own
terms via the MapBiomas Alerta platform.
