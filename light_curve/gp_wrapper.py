import george
import numpy as np

from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize


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

    def predict(self, X_pred, return_std=True):
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

    def predict(self, X_pred, return_std=True):
        pred, pred_var = self.gp.predict(self.y, X_pred.flatten(), return_var=True)
        if return_std:
            return pred, np.sqrt(pred_var)
        else:
            return pred
