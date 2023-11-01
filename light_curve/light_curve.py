import numpy as np
import numpy.typing as npt
import pandas as pd

from astropy.stats import biweight_location, biweight_scale
from feets import FeatureSpace
from feets.preprocess import remove_noise
from PyAstronomy.pyasl import foldAt
from typing import Optional

import light_curve.feets_patch
from light_curve.gp_wrapper import GPWrapper, GeorgeGPWrapper


class LightCurve:
    """A class for manipulating light curves using hjd and periodic representations simulataneously."""

    def __init__(
        self,
        lc: pd.DataFrame,
        period_catalog: float,
        id: int = -1,
        model: Optional[GPWrapper] = None,
        seed: int = 999,
    ):
        self.period_catalog = period_catalog
        self.id = id

        if model is None:
            self.model = GeorgeGPWrapper()
        else:
            self.model = model

        self.rng = np.random.default_rng(seed)

        self.hjd = lc.hjd.to_numpy()
        self.mag = lc.mag.to_numpy()
        self.err = lc.err.to_numpy()
        self.synth = np.full_like(self.hjd, False, dtype=bool)
        self.dirty = True

        self.background_level = biweight_location(self.mag)
        self.fs = FeatureSpace(only=['PeriodLS', 'Period_fit'])
        self.period = period_catalog

        self._make_periodic()

    """Use Lomb-Scargle to obtain the period using the hjd representation."""

    def _calculate_period(self):
         _, values = self.fs.extract(time=self.hjd, magnitude=self.mag, error=self.err)
         return values[0], values[1]

    """Generate the periodic representation with the current period and the hjd representation."""

    def _make_periodic(self):
        min_idx = self.mag.argmin()
        self.phase = foldAt(self.hjd, self.period, T0=self.hjd[min_idx])
        self.background_level = biweight_location(self.mag)
        self.dirty = False

    """Train the model using the periodic representation."""

    def _train(self):
        if self.dirty:
            self.make_periodic()

        phase = self.phase.reshape(-1, 1)
        self.model.fit(phase, self.mag - self.background_level, self.err)

    """Selects a hjd for synthetic observation generation."""

    def _select_hjd(self):
        return self.rng.uniform(low=self.hjd.min(), high=self.hjd.max())

    """Returns the phase of the given hjd with the current period."""

    def _hjd_to_phase(self, hjd):
        min_idx = self.mag.argmin()
        epoch = np.floor((hjd - self.hjd[min_idx]) / self.period)
        return (hjd - self.hjd[min_idx]) / self.period - epoch

    """Adds an observation to the hjd representation."""

    def _add_hjd_observation(self, hjd, mag, err):
        self.hjd = np.append(self.hjd, hjd)
        self.mag = np.append(self.mag, mag)
        self.err = np.append(self.err, err)
        self.synth = np.append(self.synth, True)
        self.dirty = True

    """Generates a single synthetic observation."""

    def _add_single_synthetic(self):
        hjd = self._select_hjd()
        phase = self._hjd_to_phase(hjd)
        mean, std = self.model.predict(np.reshape(phase, (-1, 1)), return_std=True)
        self._add_hjd_observation(hjd, mean[0] + self.background_level, std[0])

    """
    Generates n_synthetic observations using a random hjd. pmag and perr are obtained from
    the model.
    """

    def add_synthetic(self, n_synthetic):
        self._train()
        for _ in range(n_synthetic):
            self._add_single_synthetic()

    """
    Calculates the period using the hjd representation. Using both, calculates the periodic representation.
    Returns the calculated period and the period_fit (from Lomb-Scargle).
    """

    def make_periodic(self):
        self.period, period_fit = self._calculate_period()
        self._make_periodic()
        return self.period, period_fit

    """
    Generates a subsample of the light curve of size n_sample by deleting points at random.
    Deleted points are stored in a list called "discarded".
    """

    def subsample(self, n_sample: int):
        tuple_list = [(t, m, e) for t, m, e in zip(self.hjd, self.mag, self.err)]
        tuple_list = self.rng.permutation(tuple_list)
        sample = sorted(tuple_list[:n_sample], key=lambda x: x[1])
        self.hjd, self.mag, self.err = [np.array(l) for l in zip(*sample)]
        self.dirty = True
        self.make_periodic()

    """
    Removes the observations with a signal-to-noise ratio lower than SNR.
    """

    def filter_snr(self, SNR: float):
        # 1/magnitude error = flux SNR
        filtered = np.argwhere(1.0 / self.err >= SNR)
        self.hjd = self.hjd[filtered]
        self.mag = self.mag[filtered]
        self.err = self.err[filtered]
        self.synth = self.synth[filtered]

        self.background_level = biweight_location(self.mag)
        self.dirty = True

    """
    Uses feets to remove noisy points using sigma clipping.
    """

    def filter_sigma_clipping(self):
        self.hjd, self.mag, self.err = remove_noise(self.hjd, self.mag, self.err)
        self.synth = np.full_like(self.hjd, True)
        self.background_level = biweight_location(self.mag)
        self.dirty = True

    """
    Returns the light curve as a dataframe.
    """

    def to_dataframe(self):
        return pd.DataFrame.from_dict(
            {
                "id": np.full_like(self.hjd, self.id),
                "hjd": self.hjd,
                "mag": self.mag,
                "err": self.err,
                "synthetic": self.synth
            }
        )
