import pandas as pd
import sys

from light_curve.light_curve import LightCurve
from light_curve.gp_wrapper import GeorgeGPWrapper, ScikitGPWrapper
from catalog.catalog_loader import CatalogLoader

# Parameters
n_iter = 100
n_add = 2
tile = sys.argv[1]

# Get the catalogs.
loader = CatalogLoader("../catalog")
lc_df = loader.get_lc(tile)
feature_df = loader.get_features(tile)

augmented_lc = pd.DataFrame()

for _, star in feature_df.iterrows():
    light_curve = lc_df.loc[lc_df["bm_src_id"] == star.id]
    lc = LightCurve(light_curve, star.PeriodLS, star.id, GeorgeGPWrapper())
    lc.filter_snr(20)

    if len(lc.hjd):
        for _ in range(n_iter):
            lc.add_synthetic(n_add)

        augmented_lc = augmented_lc.append(lc.to_dataframe())

augmented_lc.to_csv(f"augmented_{tile}_george_snr20.csv", index=False)
