import george
import jax
import jax.numpy as jnp
import numpy as np

from abc import ABC, abstractmethod
from jaxopt import ScipyMinimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize
from tinygp import GaussianProcess
from tinygp.kernels import Matern32

jax.config.update("jax_enable_x64", True)

class GPWrapper(ABC):
    """An abstract gaussian process regressor class."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y, yerr):
        pass

    @abstractmethod
    def predict(self, X_pred, return_std=True):
        pass


class ScikitGPWrapper(GPWrapper):
    """A wrapper for the scikit_learn gaussian process regressor."""

    def __init__(self):
        self.kernel = Matern() + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def fit(self, X, y, yerr):
        self.gp.fit(X, y)

    def predict(self, X_pred, return_std=False):
        return self.gp.predict(X_pred, return_std=return_std)


class GeorgeGPWrapper(GPWrapper):
    """A wrapper for the george.GP class."""

    def __init__(self):
        self.kernel = george.kernels.Matern32Kernel(1)
        self.gp = george.GP(self.kernel)
        self.y = np.array([])

    def fit(self, X, y, yerr):
        self.gp.compute(X, yerr)
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

    def __init__(self, log_alpha, log_l, log_jitter):
        self.log_alpha = log_alpha
        self.log_l = log_l
        self.log_jitter = log_jitter

    def fit(self, X, y, y_err):
        X, y = check_X_y(X, y)
        self.X_ = X.flatten()
        self.y_ = y
        self.y_err_ = y_err

        def build_gp(params, X):
           alpha = jnp.exp(params["log_alpha"])
           l = jnp.exp(params["log_l"])
           jitter = params["jitter"]
           kernel = alpha * alpha * Matern32(scale=l)
           return GaussianProcess(kernel, X, diag=self.y_err_**2 + jitter)

        def neg_log_likelihood(params, X, y):
            gp = build_gp(params, X)
            return -gp.log_probability(y)

        params_init = {
            "log_alpha": np.float64(self.log_alpha),
            "log_l": np.float64(self.log_l),
            "jitter": np.float64(self.log_jitter),
        }

        obj = jax.jit(jax.value_and_grad(neg_log_likelihood))
        solver = ScipyMinimize(fun=neg_log_likelihood)
        solution = solver.run(params_init, X=self.X_, y=y)
        self.gp = build_gp(solution.params, self.X_)

        self.is_fitted_ = True
        return self

    def predict(self, X_pred, return_std=False):
        check_is_fitted(self)
        X_pred = check_array(X_pred)
        pred, pred_var = self.gp.predict(self.y_, X_pred.flatten(), return_var=True)
        if return_std:
            return pred, np.sqrt(pred_var)
        else:
            return pred
