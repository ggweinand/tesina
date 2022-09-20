import carpyncho
import feets
import period_reconstruction.feets_patch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from feets.preprocess import remove_noise
from PyAstronomy.pyasl import foldAt
from sklearn.gaussian_process import GaussianProcessRegressor


class LightCurve:
    """A class for manipulating light curves using hjd and periodic representations simulataneously."""

    def __init__(self, lc, period_catalog, model, seed=999):
        self.period_catalog = period_catalog
        self.model = model

        self.rng = np.random.default_rng(seed)
        self.discarded = []

        lc.sort_values("pwp_stack_src_mag3")
        self.time, self.mag, self.err = (
            lc.pwp_stack_src_hjd.values,
            lc.pwp_stack_src_mag3.values,
            lc.pwp_stack_src_mag_err3.values,
        )
        self.fs = feets.FeatureSpace(only=["PeriodLS", "Period_fit"])
        self.period = period_catalog
        self._make_periodic()

    """Use Lomb-Scargle to obtain the period using the hjd representation."""

    def _calculate_period(self):
        _, values = self.fs.extract(self.time, self.mag, self.err)
        return values[0], values[1]

    """Generate the periodic representation with the current period and the hjd representation."""

    def _make_periodic(self):
        phases = foldAt(self.time, self.period, T0=self.time[0])
        sort = np.argsort(phases)
        self.phases, self.pmag, self.perr = phases[sort], self.mag[sort], self.err[sort]

    """Train the model using the periodic representation."""

    def _train(self):
        self.model.fit(self.phases, self.pmag)

    """Selects a hjd for synthetic observation generation."""

    def _select_hjd(self):
        return self.rng.uniform(low=self.time.min(), high=self.time.max())

    """Returns the phase of the given hjd with the current period."""

    def _hjd_to_phase(self, hjd):
        return np.absolute(hjd - self.time[0]) % self.period

    """Adds an observation to the hjd representation."""

    def _add_hjd_observation(self, hjd, mag, err):
        self.time = np.append(self.time, hjd)
        self.mag = np.append(self.mag, mag)
        self.err = np.append(self.err, err)

    """Generates a single synthetic observation."""

    def _add_single_synthetic(self):
        hjd = self._select_hjd()
        phase = self._hjd_to_phase(hjd)
        mean, std = self.model.predict(np.reshape(phase, (-1, 1)), return_std=True)
        self._add_hjd_observation(hjd, mean[0], std[0])

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
        tuple_list = [(t, m, e) for t, m, e in zip(self.time, self.mag, self.err)]
        tuple_list = self.rng.permutation(tuple_list)
        self.discarded.append(tuple_list[n_sample:])
        sample = sorted(tuple_list[:n_sample], key=lambda x: x[1])
        self.time, self.mag, self.err = [np.array(l) for l in zip(*sample)]

    """
    Removes the observations with a signal-to-noise ratio lower than SNR.
    """

    def filter_snr(self, SNR: float):
        tuple_list = [(t, m, e) for t, m, e in zip(self.time, self.mag, self.err)]
        # 1/magnitude error = flux SNR
        filtered = list(filter(lambda x: 1 / x[2] >= SNR, tuple_list))
        self.time, self.mag, self.err = [np.array(l) for l in zip(*filtered)]

    """
    Uses feets to remove noisy points using sigma clipping.
    """

    def filter_sigma_clipping(self):
        self.time, self.mag, self.err = remove_noise(self.time, self.mag, self.err)
