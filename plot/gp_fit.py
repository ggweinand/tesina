import matplotlib.pyplot as plt
import numpy as np

from filtered.filtered_loader import FilteredLoader
from catalog.catalog_loader import CatalogLoader
from light_curve.light_curve import LightCurve
from light_curve.gp_wrapper import GeorgeGPWrapper, TinyGPWrapper, ScikitGPWrapper
from PyAstronomy.pyasl import foldAt
from typing import Optional


# Given a LightCurve plots the 95% confidence interval of the GP in one phase.
def plot_gp(
    lc: LightCurve, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
):
    X_pred = np.linspace(0, 1, 100).reshape(-1, 1)

    mean_prediction, std_prediction = lc.model.predict(X_pred, return_std=True)

    mean_prediction += lc.background_level

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = plt.subplot(111)

    ax.errorbar(
        lc.phase[~lc.synth],
        lc.mag[~lc.synth],
        lc.err[~lc.synth],
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )

    # Plot synthetic points in a different color.
    ax.errorbar(
        lc.phase[lc.synth],
        lc.mag[lc.synth],
        lc.err[lc.synth],
        linestyle="None",
        color="tab:red",
        marker=".",
        markersize=10,
        label="Synthetic",
    )

    ax.plot(X_pred, mean_prediction, color="tab:green", label="Mean prediction")
    ax.fill_between(
        X_pred.ravel(),
        mean_prediction - 1. * std_prediction,
        mean_prediction + 1. * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )

    ax.invert_yaxis()
    fig.legend()
    return ax


cl = CatalogLoader("../catalog")
lc = cl.get_lc("b278")
features = cl.get_features("b278")

rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]
features = features[features["vs_type"].isin(rr_lyrae)]

# star = features.loc[features["cnt"].idxmax()]
star = features.iloc[8]
light_curve = lc[lc.id == star.id]

lc = LightCurve(light_curve, star.PeriodLS, star.id, model=TinyGPWrapper(0, 0, 0))
# lc = LightCurve(light_curve, star.PeriodLS, star.id, model=ScikitGPWrapper())
lc._train()
# lc.add_synthetic(200)

fig = plt.figure()
plot_gp(lc, fig)
print(lc.model.get_params())
# print(lc.model.gp.get_params())
# print(lc.model.gp.parameter_names)
# print(lc.model.gp.parameter_vector)
fig.tight_layout()
plt.show()
