"""
GEE step 4 — Merge the per-state SAR+GLCM and context CSVs (exported to Google
Drive by steps 2 and 3, then downloaded locally) into the single model-input
file: dados_concatenados.csv.

For each state:
  * drop duplicate CODEALERTA on both sides (keep first),
  * left-join the context columns onto the SAR+GLCM table on CODEALERTA,
then concatenate the four states.

Expected input files (in --input-dir), as exported by steps 2 & 3:
  {STATE}_SAR_GLCM.csv   and   {STATE}_context.csv
  where STATE ∈ {AMAZONAS, MATO_GROSSO, PARA, RORAIMA}

Output: config.DATA_CSV  (default: data/dados_concatenados.csv)

Usage
-----
  python gee/04_merge_features.py --input-dir /path/to/downloaded/csvs
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_CSV

ID_KEY = 'CODEALERTA'
CONTEXT_COLS = [ID_KEY, 'dist_river', 'ord_flow', 'slope', 'roughness', 'tpi']
STATES = ['AMAZONAS', 'MATO_GROSSO', 'PARA', 'RORAIMA']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Directory with the {STATE}_SAR_GLCM.csv and "
                         "{STATE}_context.csv files downloaded from Drive")
    ap.add_argument("--output", default=str(DATA_CSV),
                    help="Output CSV path (default: config.DATA_CSV)")
    args = ap.parse_args()

    print("--- Step 4: merge SAR+GLCM with context per state ---")
    parts = []
    for state in STATES:
        sar_path = os.path.join(args.input_dir, f"{state}_SAR_GLCM.csv")
        ctx_path = os.path.join(args.input_dir, f"{state}_context.csv")
        if not (os.path.exists(sar_path) and os.path.exists(ctx_path)):
            print(f"  [skip] missing file(s) for {state}")
            continue

        df_sar = pd.read_csv(sar_path).drop_duplicates(subset=[ID_KEY], keep='first')
        df_ctx = pd.read_csv(ctx_path).drop_duplicates(subset=[ID_KEY], keep='first')
        ctx_cols = [c for c in CONTEXT_COLS if c in df_ctx.columns]

        merged = pd.merge(df_sar, df_ctx[ctx_cols], on=ID_KEY, how='left')
        print(f"  {state}: SAR {df_sar.shape} + context {df_ctx.shape} → {merged.shape}")
        parts.append(merged)

    if not parts:
        raise SystemExit("No state files found; check --input-dir.")

    full = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    full.to_csv(args.output, index=False)
    print(f"\n✅ Merged dataset: {full.shape} → {args.output}")


if __name__ == "__main__":
    main()
