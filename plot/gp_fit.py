import matplotlib.pyplot as plt
import numpy as np

from filter.filtered_loader import FilteredLoader
from catalog.catalog_loader import CatalogLoader
from light_curve.light_curve import LightCurve
from light_curve.gp_wrapper import GeorgeGPWrapper, TinyGPWrapper, ScikitGPWrapper
from PyAstronomy.pyasl import foldAt
from pandas import read_csv
from typing import Optional


# Given a LightCurve plots the 95% confidence interval of the GP in one phase.
def plot_gp(
    lc: LightCurve, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
):
    X_pred = np.linspace(0, 2, 200).reshape(-1, 1)

    mean_prediction, std_prediction = lc.model.predict(X_pred, return_std=True)

    mean_prediction += lc.mean_mag
    phase = np.append(lc.phase, lc.phase + 1).reshape(-1, 1)
    mag = np.append(lc.mag, lc.mag)
    err = np.append(lc.err, lc.err)
    synth = np.append(lc.synth, lc.synth)

    print(f"std entre {std_prediction.min()} y {std_prediction.max()}")
    print(f"err entre {err.min()} y {err.max()}")

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = plt.subplot(111)

    ax.errorbar(
        phase[~synth],
        mag[~synth],
        err[~synth],
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observaciones",
    )

    # Plot synthetic points in a different color.
    #     ax.errorbar(
    #         phase[synth],
    #         mag[synth],
    #         err[synth],
    #         linestyle="None",
    #         color="tab:red",
    #         marker=".",
    #         markersize=10,
    #         label="Synthetic",
    #     )

    ax.plot(X_pred, mean_prediction, color="tab:green", label="Media del GP")
    ax.fill_between(
        X_pred.ravel(),
        mean_prediction - std_prediction,
        mean_prediction + std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"Intervalo de confianza",
    )

    ax.invert_yaxis()
    plt.title(f"tinygp ajustado a la curva de luz {lc.id}")
    plt.xlabel("Fase")
    plt.ylabel("Magnitud")
    fig.legend()
    return ax


tile = "b278"

loader = FilteredLoader("")
lc = read_csv(f"../period/rrlyr_{tile}_lc.csv", dtype=loader._lc_dtype)
features = read_csv(
    f"../period/rrlyr_{tile}_features.csv", dtype=loader._features_dtype
)

star = features.iloc[3]

lc = LightCurve(
    lc[lc["id"] == star.id], star.PeriodLS, star.id, model=TinyGPWrapper(0, 0)
)
lc._train()
# lc.add_synthetic(200)

fig = plt.figure()
plot_gp(lc, fig)
# print(lc.model.get_params())
# print(lc.model.gp.get_params())
# print(lc.model.gp.parameter_names)
# print(lc.model.gp.parameter_vector)
fig.tight_layout()
plt.show()
# plt.savefig("sklearn_overfit")
