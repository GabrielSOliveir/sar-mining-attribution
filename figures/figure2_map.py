"""
Figure 2 — spatial distribution of the validated deforestation alerts used in the study.

Panel (a) shows every alert of the modeling dataset coloured by its original
MapBiomas Alerta driver label; panel (b) shows the illegal-mining alerts alone.

NOTE ON PROVENANCE. The script that produced the figure printed in the paper was
not preserved. This is a reconstruction from the same inputs: it reproduces the
per-class counts exactly (agriculture 18,348; other 45; illegal mining 7,730) and
the same spatial content, but the framing, legend box and marker rendering differ
slightly from the printed version.

Inputs
  data/dados_concatenados.csv[.zip]  polygon geometry (.geo) and state (ESTADO)
  data/driver_labels.csv             CODEALERTA -> VPRESSAO_original, i.e. the
                                     driver label before binarization into
                                     ilegal_mining / resto
  IBGE state boundaries via geobr (downloaded on first run)

Usage
  python figures/figure2_map.py          # writes FIGURES_DIR/figure2_map.{png,pdf}
"""
import json
import os

import geobr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import shape

from config import DATA_DIR, FIGURES_DIR

STUDY = {"PA": "Pará", "AM": "Amazonas", "MT": "Mato Grosso", "RR": "Roraima"}

AGRI = ("#3b76af", "o", "Agriculture")
OTHER = ("#2ca02c", "^", "Other (urban/roads/natural/aquac.)")
MINING = ("#d62728", "o", "Illegal mining")


def load_alerts():
    """Modeling dataset joined with the pre-binarization driver label."""
    src = os.path.join(DATA_DIR, "dados_concatenados.csv")
    if not os.path.exists(src):
        src += ".zip"
    df = pd.read_csv(src, usecols=["CODEALERTA", "ESTADO", ".geo"])
    labels = pd.read_csv(os.path.join(DATA_DIR, "driver_labels.csv"))
    df = df.merge(labels, on="CODEALERTA", how="left", validate="one_to_one")
    missing = df.VPRESSAO_original.isna().sum()
    if missing:
        raise SystemExit(f"{missing} alerts have no driver label")

    geom = gpd.GeoSeries([shape(json.loads(g)) for g in df[".geo"]], crs="EPSG:4326")
    # centroids are computed in an equal-area CRS, then taken back to lon/lat
    pts = geom.to_crs("EPSG:5880").centroid.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(df.drop(columns=[".geo"]), geometry=pts, crs="EPSG:4326")


def north_arrow(ax):
    ax.annotate("N", xy=(0.055, 0.955), xytext=(0.055, 0.845),
                xycoords="axes fraction", ha="center", va="center",
                fontsize=13, fontweight="bold",
                arrowprops=dict(facecolor="black", edgecolor="black",
                                width=5, headwidth=14, headlength=12))


def draw_panel(ax, df, states, study, xlim, ylim, layers, title, legend):
    states.plot(ax=ax, facecolor="#f7f7f7", edgecolor="#c8c8c8", linewidth=0.5)
    study.plot(ax=ax, facecolor="#f2f2f2", edgecolor="#1a1a1a", linewidth=1.3)
    for mask, (colour, marker, _) in layers:
        df[mask].plot(ax=ax, color=colour, marker=marker, markersize=1.6,
                      alpha=0.55, linewidth=0)
    for _, r in study.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(STUDY[r.abbrev_state], (c.x, c.y), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#1a1a1a")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=10)
    north_arrow(ax)
    if legend:
        handles = [Line2D([0], [0], marker=m, color="none", markerfacecolor=c,
                          markeredgecolor="black", markeredgewidth=0.8,
                          markersize=9, label=f"{lab}  (n={n})")
                   for (c, m, lab), n in legend]
        ax.legend(handles=handles, loc="lower left", fontsize=9,
                  framealpha=0.95, edgecolor="#999999", borderpad=0.7,
                  labelspacing=0.6, handletextpad=0.6)


def thousands(n):
    """MDPI style: a thousands separator only from five digits upwards."""
    return f"{n:,}" if n >= 10000 else str(n)


def main():
    df = load_alerts()
    is_agri = df.VPRESSAO_original == "agriculture"
    is_min = df.VPRESSAO_original == "ilegal_mining"
    is_other = ~is_agri & ~is_min
    print(f"agriculture={is_agri.sum()}  other={is_other.sum()}  "
          f"illegal_mining={is_min.sum()}")

    states = geobr.read_state(year=2020)
    study = states[states.abbrev_state.isin(STUDY)]
    xmin, ymin, xmax, ymax = study.total_bounds
    padx, pady = 0.06 * (xmax - xmin), 0.06 * (ymax - ymin)
    xlim, ylim = (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)

    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.26))
    draw_panel(axes[0], df, states, study, xlim, ylim,
               [(is_agri, AGRI), (is_other, OTHER), (is_min, MINING)],
               "(a) Validated alerts by driver",
               [(AGRI, thousands(is_agri.sum())),
                (OTHER, thousands(is_other.sum())),
                (MINING, thousands(is_min.sum()))])
    draw_panel(axes[1], df, states, study, xlim, ylim,
               [(is_min, MINING)], "(b) Illegal-mining alerts only", None)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"figure2_map.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("saved:", out)


if __name__ == "__main__":
    main()
