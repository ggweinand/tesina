import carpyncho
import numpy as np

from light_curve_sampler import LightCurveSampler
from joblib import Parallel, delayed

import warnings
import sys

# I am so sorry.
warnings.filterwarnings("ignore")

# Parameters
tile = sys.argv[1]
snr = 20


def min_obs(lc):
    """
    Removes observations at random from a stars lightcurve until the period is
    no longer obtainable. Then returns the smalles number for which it was.
    """
    n_obs = len(lc.time)
    n_sample = n_obs
    min_obs = 1
    for n_sample in range(n_obs, min_obs, -1):
        lc.subsample(n_sample)
        lc.calculate_period()

        if np.abs(lc.period - lc.period_catalog) >= 0.01:
            break

    # If after the filtering the period is lost, I consider all original
    # observations necessary (since this is a conservative estimate).
    if n_sample == n_obs:
        return lc.n_original_obs

    return n_sample


def batch_min_obs(lc_array):
    """
    Calls min_obs on every element of lc_array and returns the result in an array.
    """
    return np.array([min_obs(lc) for lc in lc_array])


# Instance the client and get the b278 tile catalogs.
client = carpyncho.Carpyncho()
b278_lc = client.get_catalog(tile, "lc")
b278_features = client.get_catalog(tile, "features")

# Keep the observations of classified light curves.
b278_features = b278_features[b278_features["vs_type"] != ""]
b278_lc = b278_lc[b278_lc.bm_src_id.isin(b278_features.id.to_numpy())]

# Create an output file.
out_file = open(f"{tile}_.csv", "w")
out_file.write("bm_src_id,vs_type,obs_threshold\n")

# Make a LightCurveSampler object for every star in the catalog.
lc_list = []
for _, star in b278_features.iterrows():
    light_curve = b278_lc[b278_lc.bm_src_id == star.id]
    lc = LightCurveSampler(light_curve, star.PeriodLS)
    lc.filter_snr(20)
    lc_list.append(lc)
lc_array = np.array(lc_list)

# Calculate min_obs for every star in parallel, using n_jobs processes.
n_jobs = 1
with Parallel(n_jobs=n_jobs, backend="multiprocessing") as P:
    d_batch_min_obs = delayed(batch_min_obs)
    batch_list = np.array_split(lc_array, abs(n_jobs))
    results = P(d_batch_min_obs(batch) for batch in batch_list)
results = np.concatenate(results)

# Write results to file.
idx = 0
for _, star in b278_features.iterrows():
    out_file.write(f"{star.id},{star.vs_type},{results[idx]}\n")
    idx += 1
