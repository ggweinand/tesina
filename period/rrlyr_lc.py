from filtered.filtered_loader import FilteredLoader
from pandas import read_csv

import sys

# Parameters
tile = sys.argv[1]

loader = FilteredLoader("../filtered")
lc = loader.get_lc(tile)
features = read_csv(f"rrlyr_{tile}_features.csv", dtype=loader._features_dtype)

lc = lc.loc[lc["id"].isin(features["id"])]
lc.to_csv(f"rrlyr_{tile}_lc.csv", index=False)
