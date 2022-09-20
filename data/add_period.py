import carpyncho
import numpy as np
import pandas as pd

client = carpyncho.Carpyncho()
tile_list = list(client.list_tiles())[1:]

for tile in tile_list:
    catalog = client.get_catalog(tile, "features")
    df = pd.read_csv(f"min_obs_{tile}_snr20.csv")
    id_list = df.bm_src_id
    period_list = np.array([catalog[catalog["id"] == id].PeriodLS for id in id_list])
    df = df.assign(PeriodLS=period_list)
    df.to_csv(f"min_obs_{tile}_snr20.csv", index=False)