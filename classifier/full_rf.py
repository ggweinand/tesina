import numpy as np
import sys

from joblib import dump

from classifier.rf_wrapper import RFWrapper
from augmented.augmented_loader import AugmentedLoader

gp_lib = "george"
tile = "b278"
snr = 20
n_synth = sys.argv[1]
n_jobs = 10

loader = AugmentedLoader("../augmented", gp_lib, snr, n_synth)
df = loader.get_features("b278")

df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
X = df.drop(["rrlyr", "id"], axis=1)
y = df["rrlyr"]

rf = RFWrapper(n_jobs=10)
rf.fit(X, y)

result = {
    "gp_lib": gp_lib,
    "tile": tile,
    "snr": snr,
    "n_synth": n_synth,
    "rf": rf,
}

dump(result, f"rf_full_{tile}_{gp_lib}_snr{snr}_synth{n_synth}", compress=3)
