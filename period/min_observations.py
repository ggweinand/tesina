import numpy as np

from filtered.filtered_loader import FilteredLoader
from light_curve.light_curve_sampler import LightCurveSampler
from joblib import Parallel, delayed
from math import isclose

import sys

# Parameters
tile = sys.argv[1]
n_jobs = 10


def min_obs(lcs: LightCurveSampler):
    """
    Removes observations at random from a star's light curve until the period is
    no longer obtainable. Then returns the smallest number for which it was.
    """
    n_obs = len(lcs.hjd)
    n_sample = n_obs
    min_obs = 1
    for n_sample in range(n_obs, min_obs, -1):
        lcs.subsample(n_sample)
        lcs.calculate_period()

        if not isclose(lcs.period, lcs.period_catalog, rel_tol=1e-5):
            break

    return min(n_sample + 1, n_obs)


def batch_min_obs(lc_array):
    """
    Calls min_obs on every element of lc_array and returns the result in an array.
    """
    return np.array([min_obs(lc) for lc in lc_array])


# Load the DataFrames.
loader = FilteredLoader("../filtered")
lc = loader.get_lc(tile)
features = loader.get_features(tile)

# Keep the observations of RR-Lyrae stars.
rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]
features = features[features["vs_type"].isin(rr_lyrae)]
lc = lc[lc.id.isin(features.id.to_numpy())]

# Divide the data into chunks.
id_chunk = np.array_split(features.id.to_numpy(), n_jobs)

# Make the corresponding LightCurveSampler objects.
lcs_chunk = []
for ids in id_chunk:
    lcs = []
    for id in ids:
        period = features[features["id"] == id].PeriodLS
        l = lc[lc["id"] == id]
        lcs.append(LightCurveSampler(l, period, id))
    lcs_chunk.append(lcs)

# Run the function for every chunk.
result = Parallel(n_jobs=n_jobs)(delayed(batch_min_obs)(lcs) for lcs in lcs_chunk)

# Join the result and save it to a file.
result = np.concatenate(result)
features["min_obs"] = result.tolist()
features.to_csv(f"min_obs_{tile}.csv", index=False)
