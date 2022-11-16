import pandas as pd

from catalog.catalog_loader import CatalogLoader
from classifier.rf_wrapper import RFWrapper

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()
df = pd.DataFrame()

# Make a dataframe with features from all tiles.
for tile in tile_list:
    df = df.append(loader.get_features(tile))
df["vs_type"].fillna("NoClasif", inplace=True)

features = df.drop(labels="vs_type")
target = df["vs_type"]

rf = RFWrapper()
rf.fit(features, target)
rf.to_file("rf_all_orig_tiles")
