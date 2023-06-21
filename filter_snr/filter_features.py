from catalog.catalog_loader import CatalogLoader
import pandas as pd

tile = "b278"
snr = 20
loader = CatalogLoader("catalog")
feature_df = loader.get_features(tile)
lc_df = pd.read_csv(f"filtered_{tile}_lc_snr{snr}.csv")

feature_df = feature_df.loc[feature_df["id"].isin(lc_df["id"].unique())]
feature_df.to_csv(f"filtered_{tile}_features_snr{snr}.csv", index=False)
