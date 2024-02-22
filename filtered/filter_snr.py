from filtered.filtered_loader import FilteredLoader
import pandas as pd
import sys

tile = sys.argv[1]
SNR = 20
loader = FilteredLoader(".")
lc_df = loader.get_lc(tile)

# 1/magnitude error = flux SNR
filtered = lc_df[1.0 / lc_df["err"] >= SNR]

filtered.to_csv(f"filtered_{tile}_lc.csv", index=False)
