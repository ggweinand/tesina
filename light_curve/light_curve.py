import feets
import light_curve.feets_patch
import numpy as np
import pandas as pd

from astropy.stats import biweight_location
from feets.preprocess import remove_noise
from PyAstronomy.pyasl import foldAt
from typing import Optional

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
        self.discarded = []

        lc = lc.sort_values("mag")
        self.hjd, self.mag, self.err = (
            lc.hjd.values,
            lc.mag.values,
            lc.err.values,
        )
        self.orig_hjd = np.copy(self.hjd)
        self.orig_mag = np.copy(self.mag)
        self.orig_err = np.copy(self.err)
        self.synth_hjd = np.empty(0)
        self.synth_mag = np.empty(0)
        self.synth_err = np.empty(0)

        self.background_level = biweight_location(self.mag)
        self.fs = feets.FeatureSpace(only=["PeriodLS", "Period_fit"])
        self.period = period_catalog

        self._make_periodic()

    """Use Lomb-Scargle to obtain the period using the hjd representation."""

    def _calculate_period(self):
        _, values = self.fs.extract(self.hjd, self.mag, self.err)
        return values[0], values[1]

    """Generate the periodic representation with the current period and the hjd representation."""

    def _make_periodic(self):
        self._sort_by_mag()
        phase = foldAt(self.hjd, self.period, T0=self.hjd[0])
        sort = np.argsort(phase)
        self.phase, self.pmag, self.perr = phase[sort], self.mag[sort], self.err[sort]

    """Train the model using the periodic representation."""

    def _train(self):
        self.model.fit(
            self.phase.reshape(-1, 1), self.pmag - self.background_level, self.perr
        )

    """Selects a hjd for synthetic observation generation."""

    def _select_hjd(self):
        return self.rng.uniform(low=self.hjd.min(), high=self.hjd.max())

    """Returns the phase of the given hjd with the current period."""

    def _hjd_to_phase(self, hjd):
        return (np.absolute(hjd - self.hjd.min()) % self.period) / self.period

    """Adds an observation to the hjd representation."""

    def _add_hjd_observation(self, hjd, mag, err):
        self.synth_hjd = np.append(self.synth_hjd, hjd)
        self.synth_mag = np.append(self.synth_mag, mag)
        self.synth_err = np.append(self.synth_err, err)
        self.hjd = np.append(self.hjd, hjd)
        self.mag = np.append(self.mag, mag)
        self.err = np.append(self.err, err)

    """Generates a single synthetic observation."""

    def _add_single_synthetic(self):
        hjd = self._select_hjd()
        phase = self._hjd_to_phase(hjd)
        mean, std = self.model.predict(np.reshape(phase, (-1, 1)), return_std=True)
        self._add_hjd_observation(hjd, mean[0] + self.background_level, std[0])

    """
    Sorts observations by ascending magnitude.
    """

    def _sort_by_mag(self):
        tuple_list = [(t, m, e) for t, m, e in zip(self.hjd, self.mag, self.err)]
        sorted_tuples = sorted(tuple_list, key=lambda x: x[1])
        self.hjd, self.mag, self.err = [np.array(l) for l in zip(*sorted_tuples)]

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
        self.discarded.append(tuple_list[n_sample:])
        sample = sorted(tuple_list[:n_sample], key=lambda x: x[1])
        self.hjd, self.mag, self.err = [np.array(l) for l in zip(*sample)]
        self.make_periodic()

    """
    Removes the observations with a signal-to-noise ratio lower than SNR.
    """

    def filter_snr(self, SNR: float):
        tuple_list = [(t, m, e) for t, m, e in zip(self.hjd, self.mag, self.err)]
        # 1/magnitude error = flux SNR
        filtered = list(filter(lambda x: 1 / x[2] >= SNR, tuple_list))
        if filtered:
            self.hjd, self.mag, self.err = [np.array(l) for l in zip(*filtered)]
        else:
            self.hjd, self.mag, self.err = [], [], []

        self.orig_hjd = np.copy(self.hjd)
        self.orig_mag = np.copy(self.mag)
        self.orig_err = np.copy(self.err)

    """
    Uses feets to remove noisy points using sigma clipping.
    """

    def filter_sigma_clipping(self):
        self.hjd, self.mag, self.err = remove_noise(self.hjd, self.mag, self.err)
        self.orig_hjd = np.copy(self.hjd)
        self.orig_mag = np.copy(self.mag)
        self.orig_err = np.copy(self.err)

    """
    Returns the light curve as a dataframe.
    """

    def to_dataframe(self):
        total_len = len(self.hjd)
        return pd.DataFrame.from_dict(
            {
                "id": np.full(total_len, self.id),
                "hjd": np.append(self.orig_hjd, self.synth_hjd),
                "mag": np.append(self.orig_mag, self.synth_mag),
                "err": np.append(self.orig_err, self.synth_err),
                "synthetic": np.append(
                    np.full(len(self.orig_hjd), False),
                    np.full(len(self.synth_hjd), True),
                ),
            }
        )
