import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# Parameters.
tile = sys.argv[1]

df = pd.read_csv(f"../period/min_obs_{tile}.csv")

ax = sns.scatterplot(data=df, x="cnt", y="min_obs", palette="deep")
ax.set_xlabel("Original amount of observations")
ax.set_ylabel("Min. amount of observations")
plt.show()
# plt.savefig(f"min_obs_{tile}.csv", bbox_inches="tight")
