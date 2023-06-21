from light_curve.light_curve_sampler import LightCurveSampler
from catalog.catalog_loader import CatalogLoader
import pandas as pd

tile = "b278"
snr = 20
loader = CatalogLoader("catalog")
lc_df = loader.get_lc(tile)
feature_df = loader.get_features(tile)

filtered_list = []
for idx, star in feature_df.iterrows():
    light_curve = lc_df.loc[lc_df["bm_src_id"] == star.id]
    lc = LightCurveSampler(light_curve, star.PeriodLS, star.id)
    lc.filter_snr(snr)

    if len(lc.hjd):
        filtered_list.append(lc.to_dataframe())

filtered_lc = pd.concat(filtered_list)
filtered_lc.to_csv(f"filtered_{tile}_lc_snr20.csv", index=False)
