import numpy as np
import h5py
import matplotlib.pyplot as plt

models = ["familiar", "unfamiliar", "familiar-unfamiliar", "face", "contour", "face-contour"]

corrs = np.array(h5py.File("../../dcnn-model/corrs/corrs.h5", "r")["corrs"])

corrs_random = np.array(h5py.File("../../dcnn-model/corrs/random_corrs.h5", "r")["corrs"])

x = np.arange(0, 8, 1)

for i in range(6):

    fig = plt.gcf()
    fig.set_size_inches(6.4, 2.5)

    plt.plot(x, corrs_random[i], "-", c="lightseagreen", lw=4, alpha=0.6)
    plt.plot(x, corrs_random[i], "o", c="lightseagreen", ms=12)

    plt.plot(x, corrs[i], "-", c="#1f77b4", lw=4, alpha=0.6)
    plt.plot(x, corrs[i], "o", c="#1f77b4", ms=12)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_position(("data", -0.5))
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_position(("data", 0))

    plt.xlim(-0.5, 7.5)
    plt.xticks(x, ["ly2", "ly4", "ly6", "ly8", "ly10", "ly12", "ly14", "ly16"], fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("\nbeta estimate (a.u.)", fontsize=15)
    figpath = "results/" + models[i] + ".jpg"
    plt.savefig(figpath, dpi=2000)
    plt.close()