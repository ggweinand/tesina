from filtered.filtered_loader import FilteredLoader
import sys

tile = sys.argv[1]
filtered_loader = FilteredLoader(".")
features = filtered_loader.get_features(tile)
lc = filtered_loader.get_lc(tile)

lc = lc.loc[lc["id"].isin(features["id"])]
lc.to_csv(f"filtered_{tile}_lc.csv", index=False)
