import feets
import gc
import light_curve.feets_patch
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from augmented.augmented_loader import AugmentedLoader
from filtered.filtered_loader import FilteredLoader

# Parameters.
gp_lib = "george"
tile = "b278"
snr = 20
n_synth = 200
n_jobs = 10


def generate_features(lc, features, fs):
    features_list = []
    for _, star in features.iterrows():
        try:
            star_lc = lc.loc[lc["id"] == star.id]
            names, values = fs.extract(
                star_lc["hjd"].values, star_lc["mag"].values, star_lc["err"].values
            )
            lc_dict = dict(zip(names, values))
            lc_dict["id"] = [star.id]
            lc_dict["rrlyr"] = star.rrlyr

            star_df = pd.DataFrame.from_dict(lc_dict)
            features_list.append(star_df)
        except RuntimeError:
            print(f"Caught RuntimeError with star id: {star.id}")
    return pd.concat(features_list)


filtered_loader = FilteredLoader("../filtered")
augmented_loader = AugmentedLoader(".", gp_lib, snr, n_synth)

# List of features to keep, plus the target (vs_type). Based on:
# Cabral, J. B., Ramos, F., Gurovich, S., & Granitto, P. M. (2020).
# Automatic catalog of RR Lyrae from ∼14 million VVV light curves:
# How far can we go with traditional machine-learning?.
# Astronomy & Astrophysics, 642, A58.
columns = [
    "Amplitude",
    "Autocor_length",
    "Beyond1Std",
    "Con",
    "Eta_e",
    "FluxPercentileRatioMid20",
    "FluxPercentileRatioMid35",
    "FluxPercentileRatioMid50",
    "FluxPercentileRatioMid65",
    "FluxPercentileRatioMid80",
    "Freq1_harmonics_amplitude_0",
    "Freq1_harmonics_amplitude_1",
    "Freq1_harmonics_amplitude_2",
    "Freq1_harmonics_amplitude_3",
    "Freq1_harmonics_rel_phase_1",
    "Freq1_harmonics_rel_phase_2",
    "Freq1_harmonics_rel_phase_3",
    "Freq2_harmonics_amplitude_0",
    "Freq2_harmonics_amplitude_1",
    "Freq2_harmonics_amplitude_2",
    "Freq2_harmonics_amplitude_3",
    "Freq2_harmonics_rel_phase_1",
    "Freq2_harmonics_rel_phase_2",
    "Freq2_harmonics_rel_phase_3",
    "Freq3_harmonics_amplitude_0",
    "Freq3_harmonics_amplitude_1",
    "Freq3_harmonics_amplitude_2",
    "Freq3_harmonics_amplitude_3",
    "Freq3_harmonics_rel_phase_1",
    "Freq3_harmonics_rel_phase_2",
    "Freq3_harmonics_rel_phase_3",
    "Gskew",
    "LinearTrend",
    "MaxSlope",
    "Mean",
    "MedianAbsDev",
    "MedianBRP",
    "PairSlopeTrend",
    "PercentAmplitude",
    "PercentDifferenceFluxPercentile",
    "PeriodLS",
    "Period_fit",
    "Psi_CS",
    "Psi_eta",
    "Q31",
    "Rcs",
    "Skew",
    "SmallKurtosis",
    "Std",
]

rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]

tile_features = filtered_loader.get_features(tile, snr)
tile_features = tile_features.assign(rrlyr=tile_features["vs_type"].isin(rr_lyrae))
tile_lc = augmented_loader.get_lc(tile)

tile_id = tile_lc["id"].unique()
chunk_id = np.array_split(tile_id, n_jobs)
lc_chunks = [tile_lc[tile_lc["id"].isin(ids)] for ids in chunk_id]
f_chunks = [tile_features[tile_features["id"].isin(ids)] for ids in chunk_id]
fs_list = [feets.FeatureSpace(only=columns)] * n_jobs

del tile_features
del tile_lc
del tile_id
del chunk_id
gc.collect()

features_list = Parallel(n_jobs=n_jobs)(
    delayed(generate_features)(lc_chunk, f_chunk, fs)
    for lc_chunk, f_chunk, fs in zip(lc_chunks, f_chunks, fs_list)
)

features_df = pd.concat(features_list)
features_df.to_csv(f"augmented_{tile}_{gp_lib}_features_snr{snr}_synth{n_synth}.csv", index=False)
