import pandas as pd

from catalog.catalog_loader import CatalogLoader

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()
df = pd.DataFrame()

# List of features to keep, plus the target (vs_type) and id. Based on:
# Cabral, J. B., Ramos, F., Gurovich, S., & Granitto, P. M. (2020).
# Automatic catalog of RR Lyrae from ∼14 million VVV light curves: How far can we go with traditional machine-learning?.
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
    "id",
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
    "c89_c3",
    "c89_hk_color",
    "c89_jh_color",
    "c89_jk_color",
    "c89_m2",
    "c89_m4",
    "n09_c3",
    "n09_hk_color",
    "n09_jh_color",
    "n09_jk_color",
    "n09_m2",
    "n09_m4",
    "ppmb",
    "vs_type",
]

# Make a dataframe with features from all tiles.
for tile in tile_list:
    tile_df = loader.get_features(tile)[columns]
    df = df.append(tile_df)

# Make a binary target (it either is a RR-Lyrae or not).
rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]
df = df.assign(rrlyr=df["vs_type"].isin(rr_lyrae))
df = df.drop(labels="vs_type", axis=1)

df.to_csv("features_all.csv", index=False)
