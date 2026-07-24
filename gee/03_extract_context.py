"""
GEE step 3 — Extract terrain / hydrology context features for the validated
master sample, and export one CSV per state to Google Drive.

Per polygon (mean reducer):
  * dist_river : distance (m) to the nearest free-flowing river
                 (WWF/HydroSHEDS/v1/FreeFlowingRivers), capped at 900 m.
  * ord_flow   : river flow order at the polygon (missing → 1).
  * slope      : terrain slope from NASADEM.
  * roughness  : local std-dev of elevation (3×3 window).
  * tpi        : Topographic Position Index (elevation − 5×5 focal mean).

Requires the validated asset produced by gee/01_explore_and_sample.py.

Usage
-----
  export GEE_PROJECT=ee-your-project
  python gee/03_extract_context.py
"""

import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import require_gee_project

PROJECT = require_gee_project()
VALIDATED_ASSET = f"projects/{PROJECT}/assets/master_sample_validated"
DRIVE_FOLDER = "SAR_features"
STATES = ['AMAZONAS', 'MATO GROSSO', 'PARÁ', 'RORAIMA']

DEM = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
RIVERS = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")


def context_stack():
    river_img = (RIVERS
                 .map(lambda f: f.set('ORD_FLOW',
                                      ee.Algorithms.If(f.get('ORD_FLOW'), f.get('ORD_FLOW'), 1)))
                 .reduceToImage(properties=['ORD_FLOW'], reducer=ee.Reducer.first())
                 .rename('ord_flow'))

    dist_river = (river_img.gt(0)
                  .fastDistanceTransform(900, 'meters')
                  .sqrt()
                  .rename('dist_river'))
    slope = ee.Terrain.slope(DEM).rename('slope')
    rough = DEM.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.square(3)).rename('roughness')
    tpi = DEM.subtract(DEM.focal_mean(5)).rename('tpi')
    return dist_river.addBands([river_img, slope, rough, tpi])


def main():
    ee.Authenticate()
    ee.Initialize(project=PROJECT)
    print("--- Step 3: terrain / hydrology context feature extraction ---")
    validated = ee.FeatureCollection(VALIDATED_ASSET)
    print(f"    Validated sample: {validated.size().getInfo()} features")

    stack = context_stack()
    for state in STATES:
        subset = validated.filter(ee.Filter.eq('ESTADO', state))
        features = stack.reduceRegions(
            collection=subset, reducer=ee.Reducer.mean(), scale=30, tileScale=4,
        )
        state_safe = state.replace('Á', 'A').replace(' ', '_')
        task = ee.batch.Export.table.toDrive(
            collection=features,
            description=f'extract_context_{state_safe}',
            folder=DRIVE_FOLDER,
            fileNamePrefix=f'{state_safe}_context',
            fileFormat='CSV',
        )
        task.start()
        print(f"✅ Context export task started → {state}")

    print("\nAll context tasks started. Monitor at "
          "https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
