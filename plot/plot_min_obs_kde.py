import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Parameters.
n_bins = 8
hue = "vs_type"

df = pd.read_csv("min_obs_all_snr20.csv")
# Descomentar para hacer el plot por vs_type.
df = df.groupby("vs_type").filter(lambda x: len(x) > 100)

# Parchazo para period_interval.
# bin_labels = [
#     "(0.099, 0.118]",
#     "(0.118, 0.152]",
#     "(0.152, 0.199]",
#     "(0.199, 0.261]",
#     "(0.261, 0.364]",
#     "(0.364, 0.54]",
#     "(0.54, 1.01]",
#     "(1.01, 200.0]",
# ]
# 
# intervals = [bin_labels[idx] for idx in pd.qcut(df.PeriodLS, n_bins, labels=False)]
# df = df.assign(period_interval=intervals)

# Parchazo para hjd_interval.
# bin_labels = [
#     "(0.099, 0.118]",
#     "(0.118, 0.152]",
#     "(0.152, 0.199]",
#     "(0.199, 0.261]",
#     "(0.261, 0.364]",
#     "(0.364, 0.54]",
#     "(0.54, 1.01]",
#     "(1.01, 200.0]",
# ]
# 
# intervals = [bin_labels[idx] for idx in pd.qcut(df.hjd_median/df.PeriodLS, n_bins, labels=False)]
# df = df.assign(hjd_interval=intervals)

# pd.qcut(df.hjd_median/df.PeriodLS, n_bins, labels=False)

# Parchazo para n_del.
# ax = sns.kdeplot(data=df, x="cnt", hue=hue, palette="deep", legend=True)
# ax.set_xlim(left=-2)
# # ax.set_xlim(right=30)
# ax.set_xlabel("Points in light curve")
# ax.set_ylabel("Density")
# plt.savefig(f"cnt_{hue}", bbox_inches="tight")

# ax = sns.kdeplot(data=df, x="obs_threshold", hue=hue, palette="deep", legend=True)
# ax.set_xlim(left=-10)
# ax.set_xlabel("Min. amount of observations")
# ax.set_ylabel("Density")
# plt.savefig(f"min_obs_{hue}", bbox_inches="tight")
