import feets
import light_curve.feets_patch
import numpy as np


class LightCurveSampler:
    """A class for generating subsamples of light curves."""

    def __init__(self, lc, period_catalog, id=-1, seed=999):
        self.id: int = id
        self.period_catalog = period_catalog
        self.period = period_catalog

        # Randomly shuffle the observations.
        self.rng = np.random.default_rng(seed)
        tuple_list = np.array(
            [
                (t, m, e)
                for t, m, e in zip(
                    lc.hjd.to_numpy(),
                    lc.mag.to_numpy(),
                    lc.err.to_numpy(),
                )
            ]
        )
        self.rng.shuffle(tuple_list)
        self.hjd, self.mag, self.err = [np.array(l) for l in zip(*tuple_list)]

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
