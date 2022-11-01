import pandas as pd

from light_curve.light_curve import LightCurve
from catalog.catalog_loader import CatalogLoader

# Get the catalogs.
loader = CatalogLoader("../catalog")
b278_lc = loader.get_lc("b278")
b278_features = loader.get_features("b278")

# Filter RRLyraes from the catalogs.
rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]
b278_features = b278_features[b278_features["vs_type"].isin(rr_lyrae)]
b278_lc = b278_lc[b278_lc.bm_src_id.isin(b278_features.id.to_numpy())]

min_obs_df = pd.read_csv("../data/min_obs_b278_snr20.csv")

n_success = 0
n_stars = 0
file = open("periodls_result.txt", "w")
file.write("id,PeriodLS,reconstructed,period_fit\n")

# Parameters.
n_iter = 20
n_synthetic = 1

for _, star in b278_features.iterrows():
    light_curve = b278_lc[b278_lc.bm_src_id == star.id]
    min_obs = min_obs_df.loc[min_obs_df.bm_src_id == star.id, "obs_threshold"].item()
    lc = LightCurve(light_curve, star.PeriodLS)
    lc.filter_snr(20)
    if len(lc.hjd) > 5:
        n_stars += 1
        lc.subsample(max(min_obs - 5, 5))
        for _ in range(n_iter):
            lc.period = lc.period_catalog
            lc._make_periodic()
            lc.add_synthetic(n_synthetic)
        period, period_fit = lc.make_periodic()
        file.write(f"{star.id},{star.PeriodLS},{period},{period_fit}\n")

        if period == lc.period_catalog:
            n_success += 1

file.write(f"Funciono en {n_success} de {n_stars}.")
