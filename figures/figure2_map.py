"""
Figure 2 — publication map of the validated alert polygons across the four study
states (two stacked panels: all validated alerts; illegal mining only).

Reads config.DATA_CSV (uses the .geo and VPRESSAO columns) and writes
figura2_mapa.{pdf,png} to config.FIGURES_DIR (override with --output-dir).

Note: downloads Brazilian state boundaries via `geobr` (needs internet on first run).

Usage
-----
  python figures/figure2_map.py
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import shape
import geobr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_CSV, FIGURES_DIR

warnings.filterwarnings("ignore")

CRS_M = 5880  # SIRGAS 2000 / Brazil Polyconic (metric)
STUDY = {"PA": "Pará", "AM": "Amazonas", "MT": "Mato Grosso", "RR": "Roraima"}


def north_arrow(ax):
    ax.annotate("N", xy=(0.06, 0.97), xytext=(0.06, 0.86),
                xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=4, headwidth=12),
                ha="center", va="center", fontsize=12, fontweight="bold")


def draw_panel(ax, pts, states, study, xlim, ylim, color, label, title):
    states.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#bdbdbd", linewidth=0.4)
    study.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=1.1)
    pts.plot(ax=ax, color=color, markersize=1.2, alpha=0.45, linewidth=0)
    for _, r in study.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(STUDY[r["abbrev_state"]], (c.x, c.y), ha="center", va="center",
                    fontsize=8, fontweight="bold", color="#222222")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.add_artist(ScaleBar(1, units="m", location="lower right",
                           box_alpha=0.7, length_fraction=0.25,
                           font_properties={"size": 7}))
    north_arrow(ax)
    leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                  markersize=6, alpha=0.8, label=label)]
    ax.legend(handles=leg, loc="lower left", fontsize=8, framealpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_CSV))
    ap.add_argument("--output-dir", default=str(FIGURES_DIR))
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Input file not found: {args.data}\nSee data/README.md.")
    os.makedirs(args.output_dir, exist_ok=True)

    print("1) Loading CSV and building geometries ...")
    df = gpd.pd.read_csv(args.data)
    geoms, bad = [], 0
    for raw in df[".geo"]:
        try:
            geoms.append(shape(json.loads(raw)))
        except Exception:
            geoms.append(None); bad += 1
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    print(f"   polygons: {len(gdf)}  | invalid geometries: {bad}")

    gdf_m = gdf.to_crs(epsg=CRS_M)
    gdf_m["geometry"] = gdf_m.geometry.centroid

    print("2) Downloading state boundaries (geobr) ...")
    states = geobr.read_state(year=2020).to_crs(epsg=CRS_M)
    study = states[states["abbrev_state"].isin(STUDY.keys())].copy()

    minx, miny, maxx, maxy = study.total_bounds
    mx, my = (maxx - minx) * 0.04, (maxy - miny) * 0.04
    xlim = (minx - mx, maxx + mx)
    ylim = (miny - my, maxy + my)

    pts_all = gdf_m
    pts_min = gdf_m[gdf_m["VPRESSAO"] == "ilegal_mining"]
    print(f"   points: all={len(pts_all)} | illegal_mining={len(pts_min)}")

    print("3) Rendering figure ...")
    w_in = 16.0 / 2.54
    fig, axes = plt.subplots(2, 1, figsize=(w_in, w_in * 1.25))
    draw_panel(axes[0], pts_all, states, study, xlim, ylim, "#d62728",
               "Validated alerts (illegal mining + other)", "(a) All validated alerts")
    draw_panel(axes[1], pts_min, states, study, xlim, ylim, "#1f77b4",
               "Illegal mining", "(b) Illegal mining only")
    plt.tight_layout()

    pdf = os.path.join(args.output_dir, "figura2_mapa.pdf")
    png = os.path.join(args.output_dir, "figura2_mapa.png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Saved: {pdf}\n         {png}")


if __name__ == "__main__":
    main()
