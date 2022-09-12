import numpy as np
import h5py
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5, 6, 7])

x = x - 0.1
sim = np.loadtxt("../../dcnn/classification_results/acc.txt")[:, 0]
avg = np.average(sim)
plt.plot(x, sim, "-", c="salmon", lw=3, label="familiar vs. unfamiliar ")
plt.plot(x, sim, "o", c="salmon", ms=6)

x = x + 0.1
sim = np.loadtxt("../../dcnn/classification_results/acc.txt")[:, 1]
avg = np.average(sim)
plt.plot(x, sim, "-", c="lightgreen", lw=3, label="familiar vs. scrambled")
plt.plot(x, sim, "o", c="lightgreen", ms=6)

x = x + 0.1
sim = np.loadtxt("../../dcnn/classification_results/acc.txt")[:, 2]
avg = np.average(sim)
plt.plot(x, sim, "-", c="skyblue", lw=3, label="unfamiliar vs. scrambled")
plt.plot(x, sim, "o", c="skyblue", ms=6)

x = x - 0.2
sim = np.loadtxt("../../dcnn/classification_results/acc_random.txt")[:, 0]
avg = np.average(sim)
plt.plot(x, sim, "--", c="salmon", lw=3, alpha=0.7, label="familiar vs. unfamiliar(random)")
plt.plot(x, sim, "o", c="salmon", ms=6, alpha=0.7)

x = x + 0.1
sim = np.loadtxt("../../dcnn/classification_results/acc_random.txt")[:, 1]
avg = np.average(sim)
plt.plot(x, sim, "--", c="lightgreen", lw=3, alpha=0.7, label="familiar vs. scrambled(random)")
plt.plot(x, sim, "o", c="lightgreen", ms=6, alpha=0.7, )

x = x + 0.1
sim = np.loadtxt("../../dcnn/classification_results/acc_random.txt")[:, 2]
avg = np.average(sim)
plt.plot(x, sim, "--", c="skyblue", lw=3, alpha=0.7, label="unfamiliar vs. scrambled(random)")
plt.plot(x, sim, "o", c="skyblue", ms=6, alpha=0.7, )

fig = plt.gcf()
fig.set_size_inches(6.5, 8)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["left"].set_position(("data", -0.4))
ax.spines["bottom"].set_linewidth(2)
ax.spines["bottom"].set_position(("data", 0.5))

#plt.ylim(-0.02, 0.82)
plt.xlim(-0.5, 7.5)
plt.ylim(0.5, 1.05)
plt.xticks(x, ["ly2", "ly4", "ly6", "ly8", "ly10", "ly12", "ly14", "ly16"], fontsize=14)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=14)
plt.legend(bbox_to_anchor=(0.35, 0.84))
leg = ax.get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize="14")
plt.ylabel("Classification Accuracy", fontsize=16)
figpath = "decoding/accs.jpg"
plt.savefig(figpath, dpi=2000)

plt.show()