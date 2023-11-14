import george
import jax
import jax.numpy as jnp
import numpy as np
import tinygp

from abc import ABC, abstractmethod
from jaxopt import ScipyMinimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

class GPWrapper(ABC):
    """An abstract gaussian process regressor class."""

    @abstractmethod
    def __init__(self, length_scale, period):
        pass

    @abstractmethod
    def fit(self, X, y, y_err):
        pass

    @abstractmethod
    def predict(self, X_pred, return_std=True):
        pass


class ScikitGPWrapper(GPWrapper, BaseEstimator, RegressorMixin):
    """A wrapper for the scikit_learn gaussian process regressor."""

    def __init__(self, scale, length_scale):
        self.scale = scale
        self.length_scale = length_scale
        self.period = 1

    def fit(self, X, y, y_err):
        self.mag_mean = np.mean(y)
        y = y - np.mean(y)
        self.err_mean = np.mean(y_err)
        y_err = y_err / self.err_mean
        print(y_err)

        kernel = np.exp(self.scale) * ExpSineSquared(np.exp(self.length_scale), self.period, periodicity_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2)
        self.gp.fit(X, y)

        params = self.gp.kernel_.theta
        self.scale = params[0] / 2
        self.length_scale = params[1]
        self.is_fitted_ = True

        return self.gp

    def predict(self, X_pred, return_std=False):
        check_is_fitted(self)
        mean, std = self.gp.predict(X_pred, return_std=return_std)
        return self.mag_mean + mean, self.err_mean/std


class GeorgeGPWrapper(GPWrapper):
    """A wrapper for the george.GP class."""

    def __init__(self, scale, length_scale):
        self.kernel = np.exp(scale) * george.kernels.Matern32Kernel(1)
        self.gp = george.GP(self.kernel)
        self.y = np.array([])

    def fit(self, X, y, y_err):
        self.gp.compute(X, y_err)
        self.y = y

        def neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.log_likelihood(y)

        def grad_neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.grad_log_likelihood(y)

        result = minimize(
            neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like
        )

        self.gp.set_parameter_vector(result.x)

    def predict(self, X_pred, return_std=False):
        pred, pred_var = self.gp.predict(self.y, X_pred.flatten(), return_var=True)
        if return_std:
            return pred, np.sqrt(pred_var)
        else:
            return pred

class TinyGPWrapper(GPWrapper, BaseEstimator, RegressorMixin):
    """A wrapper for the tinygp.GaussianProcess class."""

    def __init__(self, scale, length_scale):
        self.scale = scale
        self.length_scale = length_scale
        self.period = 1

    def fit(self, X, y, y_err):
        X, y = check_X_y(X, y)
        self.X_ = X.flatten()
        self.y_ = y
        self.y_err_ = y_err

        def build_gp(params):
           scale = jnp.exp(params["log_scale"])
           scale = 1
           length_scale = jnp.exp(params["log_length_scale"])
           kernel = scale * scale * tinygp.kernels.ExpSineSquared(gamma=length_scale, scale=self.period)
           return tinygp.GaussianProcess(kernel, self.X_, diag=self.y_err_**2)

        @jax.jit
        def neg_log_likelihood(params):
            gp = build_gp(params)
            return -gp.log_probability(self.y_)

        params_init = {
            "log_scale": np.float64(self.scale),
            "log_length_scale": np.float64(self.length_scale),
        }

        solver = ScipyMinimize(fun=neg_log_likelihood)
        solution = solver.run(params_init)

        self.scale = solution.params["log_scale"]
        self.length_scale = solution.params["log_length_scale"]

        self.gp = build_gp(solution.params)
        self.is_fitted_ = True
        return self

    def predict(self, X_pred, return_std=False):
        check_is_fitted(self)
        X_pred = check_array(X_pred)
        pred, pred_var = self.gp.predict(self.y_, X_pred.flatten(), return_var=True)
        print(f"var entre {pred_var.min()} y {pred_var.max()}")
        if return_std:
            return pred, np.sqrt(pred_var)
        else:
            return pred
