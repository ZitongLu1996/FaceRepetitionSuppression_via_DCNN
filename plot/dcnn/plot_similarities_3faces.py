import numpy as np
import h5py
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5, 6, 7])

sim = np.array(h5py.File("../../dcnn/similarities/familiar.h5", "r")["similarities"])
avg = np.average(sim, axis=1)
plt.plot(x, avg, "-", c="salmon", lw=3, label="familiar")
plt.plot(x, avg, "o", c="salmon", ms=6)

sim = np.array(h5py.File("../../dcnn/similarities/unfamiliar.h5", "r")["similarities"])
avg = np.average(sim, axis=1)
plt.plot(x, avg, "-", c="lightgreen", lw=3, label="unfamiliar")
plt.plot(x, avg, "o", c="lightgreen", ms=6)

sim = np.array(h5py.File("../../dcnn/similarities/scrambled.h5", "r")["similarities"])
avg = np.average(sim, axis=1)
plt.plot(x, avg, "-", c="skyblue", lw=3, label="scrambled")
plt.plot(x, avg, "o", c="skyblue", ms=6)

sim = np.array(h5py.File("../../dcnn/similarities/random_familiar.h5", "r")["similarities"])
avg = np.average(sim, axis=1)
plt.plot(x, avg, "--", c="salmon", lw=3, alpha=0.7, label="familiar(random)")
plt.plot(x, avg, "o", c="salmon", ms=6, alpha=0.7)

sim = np.array(h5py.File("../../dcnn/similarities/random_unfamiliar.h5", "r")["similarities"])
avg = np.average(sim, axis=1)
plt.plot(x, avg, "--", c="lightgreen", lw=3, alpha=0.7, label="unfamiliar(random)")
plt.plot(x, avg, "o", c="lightgreen", ms=6, alpha=0.7, )

sim = np.array(h5py.File("../../dcnn/similarities/random_scrambled.h5", "r")["similarities"])
avg = np.average(sim, axis=1)
plt.plot(x, avg, "--", c="skyblue", lw=3, alpha=0.7, label="scrambled(random)")
plt.plot(x, avg, "o", c="skyblue", ms=6, alpha=0.7, )

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["left"].set_position(("data", -0.4))
ax.spines["bottom"].set_linewidth(2)
ax.spines["bottom"].set_position(("data", 0))

plt.ylim(-0.02, 0.82)
plt.xlim(-0.5, 7.5)
plt.xticks(x, ["ly2", "ly4", "ly6", "ly8", "ly10", "ly12", "ly14", "ly16"], fontsize=14)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=14)
plt.legend(bbox_to_anchor=(0.6, 1))
leg = ax.get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize="14")
plt.ylabel("Average Similarity", fontsize=16)
figpath = "similarities/similarities.jpg"
plt.savefig(figpath, dpi=2000)

plt.show()