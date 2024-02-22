from filtered.filtered_loader import FilteredLoader
from light_curve.light_curve_sampler import LightCurveSampler
from math import isclose
from pandas import read_csv

import sys

# Parameters
tile = sys.argv[1]


def min_obs(lcs: LightCurveSampler) -> int:
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


loader = FilteredLoader("../filter")
lc = read_csv(f"rrlyr_{tile}_lc.csv", dtype=loader._lc_dtype)
features = read_csv(f"rrlyr_{tile}_features.csv", dtype=loader._features_dtype)

result = []
for star in features.itertuples():
    star_lc = lc[lc["id"] == star.id]
    lcs = LightCurveSampler(star_lc, star.PeriodLS)
    result.append(min_obs(lcs))

features["min_obs"] = result
features.to_csv(f"min_obs_{tile}.csv", index=False)
