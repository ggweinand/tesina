from pandas import read_csv


class AugmentedLoader:
    def __init__(self, path, gp_lib):
        self.path = path
        self.gp_lib = gp_lib
        self._dtype = {
            "err": "float64",
            "id": "int64",
            "hjd": "float64",
            "mag": "float64",
            "synthetic": "bool",
        }
        self._tiles = [
            "b206",
            "b214",
            "b216",
            "b220",
            "b228",
            "b234",
            "b247",
            "b248",
            "b261",
            "b262",
            "b263",
            "b264",
            "b277",
            "b278",
            "b356",
            "b360",
            "b396",
        ]

    def get_tile(self, tile):
        return read_csv(
            f"{self.path}/augmented_{tile}_{self.gp_lib}_snr20.csv", dtype=self._dtype
        )

    def list_tiles(self):
        return self._tiles
