import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

from catalog.catalog_loader import CatalogLoader

# Parameters.
hue = "vs_type"

df = pd.read_csv("min_obs_all_snr20.csv")
df = df[df["vs_type"].isin(["RRLyr-RRab", "RRLyr-RRc", "ECL-ELL"])]

loader = CatalogLoader("../catalog")
tile_list = loader.list_tiles()

out_file = open("peaks.txt", "w")

for tile in tile_list:
    tile_df = df[df["tile"] == tile]
    cnt = tile_df["cnt"].to_numpy()
    n, bins, _ = plt.hist(x=cnt, bins="auto")
    peaks, _ = find_peaks(np.concatenate(([min(n)], n, [min(n)])), height=5, width=1)
    peaks -= 1
    x_peaks = (bins[peaks] + bins[peaks + 1]) / 2
    out_file.write(f"{tile}: {x_peaks}\n")
