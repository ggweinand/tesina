import pandas as pd

from light_curve.light_curve import LightCurve
from light_curve.gp_wrapper import GeorgeGPWrapper, ScikitGPWrapper
from filtered.filtered_loader import FilteredLoader

# Parameters
n_iter = 100
n_add = 2
tile = "b278"
snr = 20

# Get the catalogs.
loader = FilteredLoader("../filtered")
lc_df = loader.get_lc(tile, snr)
feature_df = loader.get_features(tile, snr)

augmented_list = []
for _, star in feature_df.iterrows():
    light_curve = lc_df.loc[lc_df["id"] == star.id]
    lc = LightCurve(light_curve, star.PeriodLS, star.id, ScikitGPWrapper())

    for _ in range(n_iter):
        lc.add_synthetic(n_add)
    augmented_list.append(lc.to_dataframe())

augmented_df = pd.concat(augmented_list)
augmented_df.to_csv(f"augmented_{tile}_scikit_lc_snr{snr}_synth{n_iter*n_add}.csv", index=False)
