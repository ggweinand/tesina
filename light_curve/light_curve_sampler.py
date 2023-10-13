import feets
import light_curve.feets_patch
import numpy as np
import pandas as pd

from feets.preprocess import remove_noise


class LightCurveSampler:
    """A class for generating subsamples of light curves."""

    def __init__(self, lc, period_catalog, id=-1, seed=999):
        self.period_catalog = period_catalog
        self.id = id
        self.period = period_catalog

        # Randomly shuffle the observations.
        self.rng = np.random.default_rng(seed)
        tuple_list = np.array(
            [
                (t, m, e)
                for t, m, e in zip(
                    lc.pwp_stack_src_hjd.values,
                    lc.pwp_stack_src_mag3.values,
                    lc.pwp_stack_src_mag_err3.values,
                )
            ]
        )
        self.rng.shuffle(tuple_list)
        self.hjd, self.mag, self.err = [np.array(l) for l in zip(*tuple_list)]
        self.n_original_obs = len(self.hjd)

        self.fs = feets.FeatureSpace(only=["PeriodLS", "Period_fit"])

    def calculate_period(self):
        """Uses Lomb-Scargle to obtain the period."""
        _, values = self.fs.extract(self.hjd, self.mag, self.err)
        self.period = values[0]
        return values[0], values[1]

    def subsample(self, n_sample: int):
        """
        Generates a subsample of the light curve of size n_sample by deleting the last points.
        """
        self.hjd = self.hjd[:n_sample]
        self.mag = self.mag[:n_sample]
        self.err = self.err[:n_sample]

    def filter_snr(self, SNR: float):
        """
        Removes the observations with a signal-to-noise ratio lower than SNR.
        """
        tuple_list = [(t, m, e) for t, m, e in zip(self.hjd, self.mag, self.err)]
        # 1/magnitude error = flux SNR
        filtered = list(filter(lambda x: 1 / x[2] >= SNR, tuple_list))
        if filtered:
            self.hjd, self.mag, self.err = [np.array(l) for l in zip(*filtered)]
        else:
            self.hjd, self.mag, self.err = [], [], []

    def filter_sigma_clipping(self):
        """Uses feets to remove noisy points using sigma clipping."""
        self.hjd, self.mag, self.err = remove_noise(self.hjd, self.mag, self.err)

    """
    Returns the light curve as a dataframe.
    """

    def to_dataframe(self):
        total_len = len(self.hjd)
        return pd.DataFrame.from_dict(
            {
                "id": np.full(total_len, self.id),
                "hjd": self.hjd,
                "mag": self.mag,
                "err": self.err,
            }
        )
