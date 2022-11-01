import numpy as np
import pandas as pd

from catalog.catalog_loader import CatalogLoader

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()

for tile in tile_list:
    catalog = loader.get_features(tile)
    df = pd.read_csv(f"min_obs_{tile}_snr20.csv")
    if "PeriodLS" not in df.columns:
        period_list = np.array(
            [catalog[catalog["id"] == id].PeriodLS for id in df.bm_src_id]
        )
        df = df.assign(PeriodLS=period_list)
        df.to_csv(f"min_obs_{tile}_snr20.csv", index=False)
