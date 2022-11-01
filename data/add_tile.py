import pandas as pd

from catalog.catalog_loader import CatalogLoader

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()

for tile in tile_list:
    catalog = loader.get_features(tile)
    df = pd.read_csv(f"min_obs_{tile}_snr20.csv")
    if "tile" not in df.columns:
        df = df.assign(tile=tile)
        df.to_csv(f"min_obs_{tile}_snr20.csv", index=False)
