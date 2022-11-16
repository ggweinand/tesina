from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier


class RFWrapper:
    """
    Wrapper for sklearn RandomForestClassifier using the parameters from:
    Cabral, J. B., Ramos, F., Gurovich, S., & Granitto, P. M. (2020).
    Automatic catalog of RR Lyrae from ∼14 million VVV light curves: How far can we go with traditional machine-learning?.
    Astronomy & Astrophysics, 642, A58.

    The relevant quote is:
    RF: We created 500 decision trees with Information-Gain as
    metric, the maximum number of random selected features
    for each tree is the log2 of the total number of features, and
    the minimum number of observations in each leaf is 2.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=500,
            criterion="entropy",
            max_features="log2",
            min_samples_leaf=2,
            # TODO: Fijate n_jobs para entrenar en paralelo.
        )

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def to_file(self, filename: str):
        dump(value=self.model, filename=filename, compress=3)

    def from_file(self, filename: str):
        self.model = load(filename=filename)
