import numpy as np
import pandas as pd

from catalog.catalog_loader import CatalogLoader
from period_reconstruction.light_curve_sampler import LightCurveSampler

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()

for tile in tile_list:
    lc = loader.get_lc(tile)
    df = pd.read_csv(f"min_obs_{tile}_snr20.csv")
    if "median_hjd" not in df.columns:
        median_list = np.zeros(len(df.index))
        for idx, id in enumerate(df.bm_src_id):
            star_lc = lc[lc["bm_src_id"] == id]
            light_curve = LightCurveSampler(star_lc, df[df["bm_src_id"] == id].PeriodLS)
            light_curve.filter_snr(20)
            if len(light_curve.time):
                median_list[idx] = np.median(light_curve.time)
            else:
                median_list[idx] = -1

        df = df.assign(median_hjd=median_list)
        df.to_csv(f"min_obs_{tile}_snr20.csv", index=False)