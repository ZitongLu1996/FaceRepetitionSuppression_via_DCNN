import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel
from neurora import stats_cal
from neurora.rsa_plot import plot_corrs_hotmap_stats, plot_tbytsim_withstats

fig = plt.gcf()
fig.set_size_inches(7.5, 4.2)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(3)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_linewidth(3)
ax.spines['bottom'].set_position(('data', 0))
plt.ylim(0, 0.6)
plt.xlim(0, 1.5)
plt.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(1, 0.85))
leg = ax.get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize="13")
plt.xlabel("Time (s)", fontsize=16)
plt.yticks([0], [" "])
figpath = "results/emplty.jpg"
plt.savefig(figpath, dpi=2000)
plt.show()

