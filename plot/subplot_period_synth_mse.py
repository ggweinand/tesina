import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catalog.catalog_loader import CatalogLoader

fig, axs = plt.subplots(6, 3, sharey=True)
loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()
# The data goes from 0 to 40 synthetic points with a step size of 2.
n_steps = 20 + 1

for ax, tile in zip(axs.flatten(), tile_list):
    df = pd.read_csv(f"../data/period_synth_{tile}.csv")

    mse = np.zeros(n_steps)
    count = np.zeros(n_steps)
    for id in df["id"]:
        period_catalog = df.loc[
            (df["id"] == id) & (df["n_synth"] == 0), "period_ls"
        ].values[0]
        for idx, n_synth in enumerate(np.linspace(0, 40, n_steps)):
            period_row = df.loc[
                (df["id"] == id) & (df["n_synth"] == n_synth), "period_ls"
            ].values
            if len(period_row):
                mse[idx] += (period_row[0] - period_catalog) ** 2
                count[idx] += 1
            else:
                break

    mse = np.divide(mse, count)

    ax.plot(np.linspace(0, 40, 21), mse)
    # sns.histplot(
    #     data=tile_df, ax=ax, x="cnt", hue="vs_type", palette="deep", legend=True
    # )
    # ax.set_xlim(left=-2)
    # ax.set_xlim(right=300)
    ax.set_xlabel("Points added by Gaussian Process")
    ax.set_ylabel("Period MSE")
    ax.set_yscale("log")
    ax.set_title(tile)

plt.show()
