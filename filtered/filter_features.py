from catalog.catalog_loader import CatalogLoader
from filtered.filtered_loader import FilteredLoader
import sys

tile = sys.argv[1]
catalog_loader = CatalogLoader("../catalog")
filtered_loader = FilteredLoader(".")
features = catalog_loader.get_features(tile)
lc = filtered_loader.get_lc(tile)

features = features.loc[features["id"].isin(lc["id"].unique())]

vc = lc.id.value_counts()
features["cnt"] = features.apply(lambda x: vc[x["id"]], axis=1)

features.to_csv(f"filtered_{tile}_features.csv", index=False)
