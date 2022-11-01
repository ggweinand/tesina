import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from catalog.catalog_loader import CatalogLoader

# Parameters.
hue = "vs_type"

df = pd.read_csv("min_obs_all_snr20.csv")
df = df[df["vs_type"].isin(["RRLyr-RRab", "RRLyr-RRc", "ECL-ELL"])]

fig, axs = plt.subplots(6, 3, sharey=True)

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()

for ax, tile in zip(axs.flatten(), tile_list):
    tile_df = df[df["tile"] == tile]
    sns.histplot(
        data=tile_df, ax=ax, x="cnt", hue="vs_type", palette="deep", legend=True
    )
    ax.set_xlim(left=-2)
    ax.set_xlim(right=300)
    ax.set_xlabel("Points in light curve")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(tile)

plt.show()
# plt.savefig(f"sub_hist_vs_type", bbox_inches="tight")
