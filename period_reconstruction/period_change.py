import numpy as np
import pandas as pd
import sys

from light_curve.light_curve import LightCurve
from catalog.catalog_loader import CatalogLoader

tile = sys.argv[1]

# Get the catalogs.
loader = CatalogLoader("../catalog")
lc_df = loader.get_lc(tile)
features_df = loader.get_features(tile)

# Filter RRLyrae from the catalogs.
rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]
features_df = features_df[features_df["vs_type"].isin(rr_lyrae)]
lc_df = lc_df[lc_df.bm_src_id.isin(features_df.id.to_numpy())]

file = open(f"period_synth_{tile}.csv", "w")
file.write("id,n_obs,n_synth,period_catalog,period_ls,period_fit\n")

# Parameters
n_iter = 20
n_add = 2

augmented_lc = pd.DataFrame()

for _, star in features_df.iterrows():
    light_curve = lc_df[lc_df.bm_src_id == star.id]
    lc = LightCurve(light_curve, star.PeriodLS, star.id)
    lc.filter_snr(20)

    if len(lc.hjd):
        period_ls, period_fit = star.PeriodLS, star.Period_fit
        file.write(f"{lc.id},{len(lc.synth_hjd)},{period_ls},{period_fit}\n")
        for _ in range(n_iter):
            lc.add_synthetic(n_add)
            period_ls, period_fit = lc.make_periodic()
            # Ignore the lc if Lomb-Scargle returns nan.
            if np.isnan(period_ls):
                break

            file.write(f"{lc.id},{len(lc.synth_hjd)},{period_ls},{period_fit}\n")
        augmented_lc = augmented_lc.append(lc.to_dataframe())

augmented_lc.to_csv(f"augmented_{tile}_snr20.csv", index=False)
