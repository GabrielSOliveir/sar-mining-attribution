"""
GEE step 2 — Extract Sentinel-1 SAR backscatter + GLCM texture features for the
validated master sample, and export one CSV per state to Google Drive.

For each validated alert polygon:
  * Take the first Sentinel-1 IW/VV+VH DESCENDING scene within 30 days of DTIMGDEP.
  * Apply a Refined Lee speckle filter (in linear power domain, back to dB).
  * Compute VV (dB), VH (dB) and VV−VH (dB).
  * Compute GLCM texture (glcmTexture, window size 7) on scaled int16 VV/VH:
    contrast, dissimilarity (diss), IDM, ASM, correlation (corr), entropy (ent).
  * Reduce all bands over the polygon with mean, stdDev and the 10th/90th
    percentiles → the *_mean / *_stdDev / *_p10 / *_p90 feature columns.

Requires the validated asset produced by gee/01_explore_and_sample.py.

Usage
-----
  export GEE_PROJECT=ee-your-project
  python gee/02_extract_sar_glcm.py
"""

import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import require_gee_project

PROJECT = require_gee_project()
VALIDATED_ASSET = f"projects/{PROJECT}/assets/master_sample_validated"
DRIVE_FOLDER = "SAR_features"   # Google Drive output folder (configurable)
STATES = ['AMAZONAS', 'MATO GROSSO', 'PARÁ', 'RORAIMA']

GLCM_METRICS = ['contrast', 'diss', 'idm', 'asm', 'corr', 'ent']

REDUCERS = (ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True))

S1_BASE = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))


# ── Refined Lee speckle filter ────────────────────────────────────────────────
def to_natural(img):
    return ee.Image(10.0).pow(img.divide(10.0))

def to_db(img):
    return ee.Image(img).log10().multiply(10.0)

def refined_lee(image):
    def apply_filter(img_pow):
        k3 = ee.Kernel.fixed(3, 3, ee.List.repeat(ee.List.repeat(1, 3), 3), 1, 1, False)
        mean3 = img_pow.reduceNeighborhood(ee.Reducer.mean(), k3)
        k7 = ee.Kernel.fixed(7, 7, ee.List.repeat(ee.List.repeat(1, 7), 7), 3, 3, False)
        mean7 = img_pow.reduceNeighborhood(ee.Reducer.mean(), k7)
        var7 = img_pow.reduceNeighborhood(ee.Reducer.variance(), k7)
        var_noise = var7.multiply(1)
        b = var_noise.divide(var7)
        b = b.where(b.gt(1), 1)
        out = mean7.add(b.multiply(img_pow.subtract(mean7)))
        return out.where(var7.lt(var_noise), mean3)

    vv_pow = to_natural(image.select('VV'))
    vh_pow = to_natural(image.select('VH'))
    vv_db = to_db(apply_filter(vv_pow)).rename('VV')
    vh_db = to_db(apply_filter(vh_pow)).rename('VH')
    return image.addBands(vv_db, ['VV'], True).addBands(vh_db, ['VH'], True)


def extract_sar_glcm(feature):
    geom = feature.geometry()
    start = ee.Date(feature.get('DTIMGDEP'))
    end = start.advance(30, 'day')
    scene = (S1_BASE.filterBounds(geom).filterDate(start, end)
             .sort('system:time_start'))
    s1_image = ee.Image(scene.first())
    filtered = refined_lee(s1_image)

    vv_db = filtered.select('VV')
    vh_db = filtered.select('VH')
    ratio = vv_db.subtract(vh_db).rename('VV_minus_VH')

    vv_int = vv_db.multiply(1000).toInt16()
    vh_int = vh_db.multiply(1000).toInt16()
    vv_glcm = (vv_int.glcmTexture(size=7)
               .select([f'VV_{m}' for m in GLCM_METRICS])
               .rename([f'{m}_vv' for m in GLCM_METRICS]))
    vh_glcm = (vh_int.glcmTexture(size=7)
               .select([f'VH_{m}' for m in GLCM_METRICS])
               .rename([f'{m}_vh' for m in GLCM_METRICS]))

    bands = vv_db.addBands(vh_db).addBands(ratio).addBands(vv_glcm).addBands(vh_glcm)
    stats = bands.reduceRegion(
        reducer=REDUCERS, geometry=geom, scale=10,
        bestEffort=True, maxPixels=int(1e9), tileScale=4,
    )
    return feature.setMulti(stats).set({
        'scene_id': s1_image.get('system:id'),
        'scene_date': ee.Date(s1_image.get('system:time_start')).format('YYYY-MM-dd'),
    })


def main():
    ee.Authenticate()
    ee.Initialize(project=PROJECT)
    print("--- Step 2: SAR + GLCM feature extraction ---")
    validated = ee.FeatureCollection(VALIDATED_ASSET)
    print(f"    Validated sample: {validated.size().getInfo()} features")

    for state in STATES:
        subset = validated.filter(ee.Filter.eq('ESTADO', state))
        features = subset.map(extract_sar_glcm)
        state_safe = state.replace('Á', 'A').replace(' ', '_')
        task = ee.batch.Export.table.toDrive(
            collection=features,
            description=f'extract_SAR_GLCM_{state_safe}',
            folder=DRIVE_FOLDER,
            fileNamePrefix=f'{state_safe}_SAR_GLCM',
            fileFormat='CSV',
        )
        task.start()
        print(f"✅ SAR/GLCM export task started → {state}")

    print("\nAll SAR/GLCM tasks started. Monitor at "
          "https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
