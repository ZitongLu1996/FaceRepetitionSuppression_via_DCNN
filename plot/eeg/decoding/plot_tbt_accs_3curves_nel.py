import numpy as np
import h5py
from scipy.stats import ttest_1samp, ttest_rel
import matplotlib.pyplot as plt
from neurora.stuff import clusterbased_permutation_1d_1samp_1sided

nsubs = 18

accs = np.zeros([nsubs, 100, 100])
for sub in range(nsubs):
    subaccs = np.array(h5py.File("../../../eeg/decoding_acc/fn-fe_sub" + str(sub + 1) + ".h5", "r")["acc"])
    print(subaccs.shape)
    accs[sub] = subaccs
rlts1 = np.zeros([nsubs, 100])
for t in range(100):
    rlts1[:, t] = accs[:, t, t]
print(rlts1[:, 14:16])
avg = np.average(rlts1, axis=0)
err = np.zeros([100])
for t in range(100):
    err[t] = np.std(rlts1[:, t], ddof=1) / np.sqrt(nsubs)
ps = np.zeros([100])
chance = 0.5
ps = clusterbased_permutation_1d_1samp_1sided(rlts1, level=chance, p_threshold=0.01, iter=1000)
for t in range(100):
    if t >= 25:
        if ps[t] == 1:
            plt.plot(t * 0.02 + 0.01 - 0.5, 0.79, "s", color="orangered", alpha=0.9, markersize=3)
            xi = [t * 0.02 - 0.5, t * 0.02 + 0.02 - 0.5]
x = np.arange(-0.5 + 0.01, 1.5 + 0.01, 0.02)
plt.fill_between(x, avg + err, avg - err, facecolor="orangered", alpha=0.9, label="Familiar")

accs = np.zeros([nsubs, 100, 100])
for sub in range(nsubs):
    subaccs = np.array(h5py.File("../../../eeg/decoding_acc/un-ue_sub" + str(sub + 1) + ".h5", "r")["acc"])
    print(subaccs.shape)
    accs[sub] = subaccs
rlts2 = np.zeros([nsubs, 100])
for t in range(100):
    rlts2[:, t] = accs[:, t, t]
print(rlts2[:, 14:16])
avg = np.average(rlts2, axis=0)
err = np.zeros([100])
for t in range(100):
    err[t] = np.std(rlts2[:, t], ddof=1) / np.sqrt(nsubs)
ps = np.zeros([100])
chance = 0.5
ps = clusterbased_permutation_1d_1samp_1sided(rlts2, level=chance, p_threshold=0.01, iter=1000)
for t in range(100):
    if t >= 25:
        if ps[t] == 1:
            plt.plot(t * 0.02 + 0.01 - 0.5, 0.782, "s", color="orangered", alpha=0.6, markersize=3)
            xi = [t * 0.02 - 0.5, t * 0.02 + 0.02 - 0.5]
x = np.arange(-0.5 + 0.01, 1.5 + 0.01, 0.02)
plt.fill_between(x, avg + err, avg - err, facecolor="orangered", alpha=0.6, label="Unfamiliar")

accs = np.zeros([nsubs, 100, 100])
for sub in range(nsubs):
    subaccs = np.array(h5py.File("../../../eeg/decoding_acc/sn-se_sub" + str(sub + 1) + ".h5", "r")["acc"])
    print(subaccs.shape)
    accs[sub] = subaccs
rlts3 = np.zeros([nsubs, 100])
for t in range(100):
    rlts3[:, t] = accs[:, t, t]
print(rlts3[:, 14:16])
avg = np.average(rlts3, axis=0)
err = np.zeros([100])
for t in range(100):
    err[t] = np.std(rlts3[:, t], ddof=1) / np.sqrt(nsubs)
ps = np.zeros([100])
chance = 0.5
ps = clusterbased_permutation_1d_1samp_1sided(rlts3, level=chance, p_threshold=0.01, iter=1000)
for t in range(100):
    if t >= 25:
        if ps[t] == 1:
            plt.plot(t * 0.02 + 0.01 - 0.5, 0.774, "s", color="orangered", alpha=0.3, markersize=3)
            xi = [t * 0.02 - 0.5, t * 0.02 + 0.02 - 0.5]
x = np.arange(-0.5 + 0.01, 1.5 + 0.01, 0.02)
plt.fill_between(x, avg + err, avg - err, facecolor="orangered", alpha=0.3, label="Scrambled")

ps1 = clusterbased_permutation_1d_1samp_1sided(rlts1-rlts2, level=0, p_threshold=0.01, iter=1000)
ps2 = clusterbased_permutation_1d_1samp_1sided(rlts1-rlts3, level=0, p_threshold=0.01, iter=1000)
ps3 = clusterbased_permutation_1d_1samp_1sided(rlts2-rlts3, level=0, p_threshold=0.01, iter=1000)
for t in range(100):
    if t >= 25:
        if ps1[t] == 1:
            plt.plot(t * 0.02 + 0.01 - 0.5, 0.43, "s", color="b", alpha=0.9, markersize=3)
        if ps2[t] == 1:
            plt.plot(t * 0.02 + 0.01 - 0.5, 0.422, "s", color="b", alpha=0.45, markersize=3)
        if ps3[t] == 1:
            plt.plot(t * 0.02 + 0.01 - 0.5, 0.414, "s", color="b", alpha=0.2, markersize=3)

fig = plt.gcf()
fig.set_size_inches(7.2, 4)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(3)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_linewidth(3)
ax.spines["bottom"].set_position(("data", 0.5))
plt.legend(bbox_to_anchor=(1, 0.85))
leg = ax.get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize="13")
plt.ylim(0.4, 0.8)
plt.xlim(-0.25, 1.5)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("      Classification Accuracy", fontsize=16)
plt.show()