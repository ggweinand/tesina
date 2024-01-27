from catalog.catalog_loader import CatalogLoader
import sys

tile = sys.argv[1]
loader = CatalogLoader("../catalog")
feature_lc = loader.get_lc(tile)

# Dadas las caracteristicas tecnicas del relevamiento, todas las fuentes
# en los Band-Merge de magnitud <= 12 y >= 16.5 son ignoradas (Gran et
# al., 2015).
filtered = feature_lc[feature_lc["mag"].between(12, 16.5)]
filtered.to_csv(f"filtered_mag_{tile}_lc.csv", index=False)
