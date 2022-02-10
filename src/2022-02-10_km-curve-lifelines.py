# # Using lifelines to fit KM curves

# References:
#   - https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html  # noqa


from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell  # noqa

InteractiveShell.ast_node_interactivity = "all"

# ## Example 1
waltons = load_waltons()
waltons.columns = ["time", "event", "group"]
waltons.describe(include="all").T

kmf = KaplanMeierFitter(label="waltons_data")
kmf.fit(waltons["time"], waltons["event"])
kmf.plot_survival_function()
plt.show()

# ## Example 2
