import pandas as pd
import feets
import light_curve.feets_patch
import sys

from data.augmented_loader import AugmentedLoader
from catalog.catalog_loader import CatalogLoader

# Parameters.
gp_lib = sys.argv[1]
n_synth = 50

catalog_loader = CatalogLoader("../catalog")
augmented_loader = AugmentedLoader("../data", gp_lib)
tile_list = augmented_loader.list_tiles()
df = pd.DataFrame()

# List of features to keep, plus the target (vs_type). Based on:
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

# Make a dataframe with features from all tiles.
fs = feets.FeatureSpace(only=columns)
for tile in tile_list:
    tile_features = catalog_loader.get_features(tile)
    tile_features = tile_features.assign(rrlyr=tile_features["vs_type"].isin(rr_lyrae))
    tile_lc = augmented_loader.get_lc(tile)

    filtered_lc = tile_lc.loc[tile_lc["synthetic"] == False]
    for id in tile_lc["id"].unique():
        filtered_lc = filtered_lc.append(
            tile_lc.loc[(tile_lc["id"] == id) & (tile_lc["synthetic"] == True)].iloc[
                :n_synth
            ]
        )

    for id in filtered_lc["id"].unique():
        star_lc = filtered_lc.loc[filtered_lc["id"] == id]
        names, values = fs.extract(
            star_lc["hjd"].values, star_lc["mag"].values, star_lc["err"].values
        )
        
        rrlyr = tile_features[tile_features["id"] == id]["rrlyr"].values[0]
        lc_dict = dict(zip(names, values))
        lc_dict["id"] = id
        lc_dict["rrlyr"] = rrlyr
        lc_dict["tile"] = tile

        star_df = pd.DataFrame([lc_dict])
        df = df.append(star_df)

df.to_csv(f"features_{gp_lib}_{n_synth}.csv", index=False)
