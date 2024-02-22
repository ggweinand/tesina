import numpy as np
import pandas as pd

from light_curve.light_curve import LightCurve
from light_curve.gp_wrapper import GeorgeGPWrapper, ScikitGPWrapper
from filtered.filtered_loader import FilteredLoader

# Parameters
batch_idx = 0
n_batch = 10
n_jobs = 10
n_iter = 5
n_add = 2
tile = "b278"
snr = 20

# Get the catalogs.
loader = FilteredLoader("../filtered")
lc_df = loader.get_lc(tile, snr)
feature_df = loader.get_features(tile, snr)

tile_id = lc_df["id"].unique()
batch_id = np.array_split(tile_id, n_batch)
tile_id = batch_id[batch_idx]

chunk_id = np.array_split(tile_id, n_jobs)
lc_chunks = [lc_df[lc_df["id"].isin(ids)] for ids in chunk_id]
f_chunks = [feature_df[feature_df["id"].isin(ids)] for ids in chunk_id]
augmented_list = []

lc = lc_chunks[0]
f = f_chunks[0]

for _, star in f.iterrows():
    light_curve = lc.loc[lc["id"] == star.id]
    light_curve = LightCurve(light_curve, star.PeriodLS, star.id, GeorgeGPWrapper())

    for _ in range(n_iter):
        light_curve.add_synthetic(n_add)
    augmented_list.append(light_curve.to_dataframe())

augmented_df = pd.concat(augmented_list)
augmented_df.to_csv(
    f"augmented_{tile}_scikit_lc_snr{snr}_synth{n_iter*n_add}.csv", index=False
)
