from pandas import read_csv


class AugmentedLoader:
    def __init__(self, path, gp_lib):
        self.path = path
        self.gp_lib = gp_lib
        self._lc_dtype = {
            "err": "float64",
            "id": "int64",
            "hjd": "float64",
            "mag": "float64",
            "synthetic": "bool",
        }
        self._features_dtype = {
            "id": "int64",
            "tile": "str",
            "rrlyr": "bool",
            "Amplitude": "float64",
            "Autocor_length": "float64",
            "Beyond1Std": "float64",
            "Con": "float64",
            "Eta_e": "float64",
            "FluxPercentileRatioMid20": "float64",
            "FluxPercentileRatioMid35": "float64",
            "FluxPercentileRatioMid50": "float64",
            "FluxPercentileRatioMid65": "float64",
            "FluxPercentileRatioMid80": "float64",
            "Freq1_harmonics_amplitude_0": "float64",
            "Freq1_harmonics_amplitude_1": "float64",
            "Freq1_harmonics_amplitude_2": "float64",
            "Freq1_harmonics_amplitude_3": "float64",
            "Freq1_harmonics_rel_phase_0": "float64",
            "Freq1_harmonics_rel_phase_1": "float64",
            "Freq1_harmonics_rel_phase_2": "float64",
            "Freq1_harmonics_rel_phase_3": "float64",
            "Freq2_harmonics_amplitude_0": "float64",
            "Freq2_harmonics_amplitude_1": "float64",
            "Freq2_harmonics_amplitude_2": "float64",
            "Freq2_harmonics_amplitude_3": "float64",
            "Freq2_harmonics_rel_phase_0": "float64",
            "Freq2_harmonics_rel_phase_1": "float64",
            "Freq2_harmonics_rel_phase_2": "float64",
            "Freq2_harmonics_rel_phase_3": "float64",
            "Freq3_harmonics_amplitude_0": "float64",
            "Freq3_harmonics_amplitude_1": "float64",
            "Freq3_harmonics_amplitude_2": "float64",
            "Freq3_harmonics_amplitude_3": "float64",
            "Freq3_harmonics_rel_phase_0": "float64",
            "Freq3_harmonics_rel_phase_1": "float64",
            "Freq3_harmonics_rel_phase_2": "float64",
            "Freq3_harmonics_rel_phase_3": "float64",
            "Gskew": "float64",
            "LinearTrend": "float64",
            "MaxSlope": "float64",
            "Mean": "float64",
            "MedianAbsDev": "float64",
            "MedianBRP": "float64",
            "PairSlopeTrend": "float64",
            "PercentAmplitude": "float64",
            "PercentDifferenceFluxPercentile": "float64",
            "PeriodLS": "float64",
            "Period_fit": "float64",
            "Psi_CS": "float64",
            "Psi_eta": "float64",
            "Q31": "float64",
            "Rcs": "float64",
            "Skew": "float64",
            "SmallKurtosis": "float64",
            "Std": "float64",
        }
        self._tiles = [
            "b206",
            "b214",
            "b216",
            "b220",
            "b228",
            "b234",
            "b247",
            "b248",
            "b261",
            "b262",
            "b263",
            "b264",
            "b277",
            "b278",
            "b356",
            "b360",
            "b396",
        ]

    def get_lc(self, tile):
        return read_csv(
            f"{self.path}/augmented_{tile}_{self.gp_lib}_snr20.csv", dtype=self._lc_dtype
        )
    
    def get_features(self, n_synth):
        return read_csv(
            f"{self.path}/features_{self.gp_lib}_{n_synth}.csv", dtype=self._features_dtype
        )

    def list_tiles(self):
        return self._tiles
