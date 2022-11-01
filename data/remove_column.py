import pandas as pd

tiles = [
    "b206",
]
column = "avg_hjd"

for tile in tiles:
    df = pd.read_csv(f"min_obs_{tile}_snr20.csv")
    df = df.drop(columns=column)
    df.to_csv(f"min_obs_{tile}_snr20.csv", index=False)
