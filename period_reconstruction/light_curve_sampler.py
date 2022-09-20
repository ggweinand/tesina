import feets
import period_reconstruction.feets_patch
import numpy as np

from feets.preprocess import remove_noise


class LightCurveSampler:
    """A class for generating subsamples of light curves."""

    def __init__(self, lc, catalog_period, seed=999):
        self.period_catalog = catalog_period
        self.period = catalog_period

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
        self.time, self.mag, self.err = [np.array(l) for l in zip(*tuple_list)]
        self.n_original_obs = len(self.time)

        self.fs = feets.FeatureSpace(only=["PeriodLS", "Period_fit"])

    def calculate_period(self):
        """Uses Lomb-Scargle to obtain the period."""
        _, values = self.fs.extract(self.time, self.mag, self.err)
        self.period = values[0]
        return values[0], values[1]

    def subsample(self, n_sample: int):
        """
        Generates a subsample of the light curve of size n_sample by deleting the last points.
        """
        self.time = self.time[:n_sample]
        self.mag = self.mag[:n_sample]
        self.err = self.err[:n_sample]

    def filter_snr(self, SNR: float):
        """
        Removes the observations with a signal-to-noise ratio lower than SNR.
        """
        tuple_list = [(t, m, e) for t, m, e in zip(self.time, self.mag, self.err)]
        # 1/magnitude error = flux SNR
        filtered = list(filter(lambda x: 1 / x[2] >= SNR, tuple_list))
        if filtered:
            self.time, self.mag, self.err = [np.array(l) for l in zip(*filtered)]
        else:
            self.time, self.mag, self.err = [], [], []

    def filter_sigma_clipping(self):
        """Uses feets to remove noisy points using sigma clipping."""
        self.time, self.mag, self.err = remove_noise(self.time, self.mag, self.err)


class LightCurveSamplerHistory:
    """A class for generating subsamples of light curves (with history)."""

    def __init__(self, lc, catalog_period, seed=999):
        self.period_catalog = catalog_period
        self.period = catalog_period

        self.rng = np.random.default_rng(seed)
        self.discarded = []

        # Randomly shuffle the observations.
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
        self.time, self.mag, self.err = [np.array(l) for l in zip(*tuple_list)]

        self.fs = feets.FeatureSpace(only=["PeriodLS", "Period_fit"])

    def calculate_period(self):
        """Uses Lomb-Scargle to obtain the period."""
        _, values = self.fs.extract(self.time, self.mag, self.err)
        self.period = values[0]
        return values[0], values[1]

    def subsample(self, n_sample: int):
        """
        Generates a subsample of the light curve of size n_sample by deleting the last points.
        Deleted points are stored in a list called "discarded".
        """
        self.discarded.append(
            [
                (t, m, e)
                for t, m, e in zip(
                    self.time[n_sample:], self.mag[n_sample:], self.err[n_sample:]
                )
            ]
        )

        self.time = self.time[:n_sample]
        self.mag = self.mag[:n_sample]
        self.err = self.err[:n_sample]
