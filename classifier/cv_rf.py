import numpy as np
import sys

from joblib import dump
from sklearn.model_selection import StratifiedKFold

from augmented.augmented_loader import AugmentedLoader
from classifier.rf_wrapper import RFWrapper

# Parameters
gp_lib = "george"
tile = "b278"
snr = 20
n_synth = int(sys.argv[1])
n_jobs = 10
n_splits = 10

loader = AugmentedLoader("../augmented", gp_lib, snr, n_synth)
rf = RFWrapper(n_jobs=n_jobs)
skf = StratifiedKFold(n_splits=n_splits, random_state=999)

df = loader.get_features(tile)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
X = df.drop(["rrlyr", "id"], axis=1).values
y = df["rrlyr"].values

probabilities = None
predictions = np.array([])
true_labels = np.array([])
for train, test in skf.split(X, y):
    x_train = X[train]
    y_train = y[train]
    x_test = X[test]
    y_test = y[test]

    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    prob = rf.predict_proba(x_test)

    true_labels = np.hstack([true_labels, y_test])
    predictions = np.hstack([predictions, pred])
    if probabilities is None:
        probabilities = prob
    else:
        probabilities = np.vstack([probabilities, prob])

result = {
    "gp_lib": gp_lib,
    "tile": tile,
    "snr": snr,
    "n_synth": n_synth,
    "probabilities": probabilities,
    "predictions": predictions,
    "true_labels": true_labels,
}

dump(result, f"kfolds_{tile}_{gp_lib}_snr{snr}_synth{n_synth}", compress=3)
