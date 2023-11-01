from pandas import read_csv


class CatalogLoader:
    def __init__(self, path):
        self.path = path
        self._features_dtype = {
            "id": "int64",
            "cnt": "int64",
            "ra_k": "float64",
            "dec_k": "float64",
            "vs_type": "str",
            "vs_catalog": "str",
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
            "Meanvariance": "float64",
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
            "StetsonK": "float64",
            "c89_c3": "float64",
            "c89_hk_color": "float64",
            "c89_jh_color": "float64",
            "c89_jk_color": "float64",
            "c89_m2": "float64",
            "c89_m4": "float64",
            "n09_c3": "float64",
            "n09_hk_color": "float64",
            "n09_jh_color": "float64",
            "n09_jk_color": "float64",
            "n09_m2": "float64",
            "n09_m4": "float64",
            "ppmb": "float64",
        }
        self._lc_dtype = {
            "bm_src_id": "int64",
            "pwp_id": "int64",
            "pwp_stack_src_id": "int64",
            "pwp_stack_src_hjd": "float64",
            "pwp_stack_src_mag3": "float64",
            "pwp_stack_src_mag_err3": "float64",
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

    def get_features(self, tile):
        return read_csv(
            f"{self.path}/{tile}_features.csv", dtype=self._features_dtype
        ).iloc[:, 1:]

    def get_lc(self, tile):
        df = read_csv(f"{self.path}/{tile}_lc.csv", dtype=self._lc_dtype).iloc[:, 1:]
        return df.rename(
            columns={
                "bm_src_id": "id",
                "pwp_stack_src_hjd": "hjd",
                "pwp_stack_src_mag3": "mag",
                "pwp_stack_src_mag_err3": "err",
            }
        )

    def list_tiles(self):
        return self._tiles
