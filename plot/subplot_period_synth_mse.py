import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from catalog.catalog_loader import CatalogLoader

gp_lib = sys.argv[1]

fig, axs = plt.subplots(6, 3, sharey=True)
loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()
# The data goes from 0 to 200 synthetic points with a step size of 2.
n_steps = 100 + 1

for ax, tile in zip(axs.flatten(), tile_list):
    df = pd.read_csv(f"../data/period_synth_{gp_lib}_{tile}.csv")

    squared_err = np.zeros(n_steps)
    count = np.zeros(n_steps)
    for id in df["id"]:
        period_catalog = df.loc[
            (df["id"] == id) & (df["n_synth"] == 0), "period_ls"
        ].values[0]
        for idx, n_synth in enumerate(np.linspace(0, 200, n_steps)):
            period_row = df.loc[
                (df["id"] == id) & (df["n_synth"] == n_synth), "period_ls"
            ].values
            if len(period_row):
                squared_err[idx] += (period_row[0] - period_catalog) ** 2
                count[idx] += 1
            else:
                break

    X, y = [], []
    for idx, (err, cnt) in enumerate(zip(squared_err, count)):
        if cnt:
            X.append(2 * idx)
            y.append(err / cnt)

    ax.scatter(X, y)
    # sns.histplot(
    #     data=tile_df, ax=ax, x="cnt", hue="vs_type", palette="deep", legend=True
    # )
    # ax.set_xlim(left=-2)
    # ax.set_xlim(right=300)
    ax.set_ylim((10**-6, 10**2))
    ax.set_xlabel("Points added by Gaussian Process")
    ax.set_ylabel("Period MSE")
    ax.set_yscale("log")
    ax.set_title(tile)

fig.suptitle(f"Using GP from {gp_lib}")
fig.tight_layout()

plt.show()
