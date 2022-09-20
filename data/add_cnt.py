import numpy as np
import pandas as pd

from catalog.catalog_loader import CatalogLoader

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()

for tile in tile_list:
    catalog = loader.get_features(tile)
    df = pd.read_csv(f"min_obs_{tile}_snr20.csv")
    if "cnt" not in df.columns:
        id_list = df.bm_src_id
        period_list = np.array([catalog[catalog["id"] == id].cnt for id in id_list])
        df = df.assign(cnt=period_list)
        df.to_csv(f"min_obs_{tile}_snr20.csv", index=False)