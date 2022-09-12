import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel

corrs1 = np.zeros([18, 8, 100, 2])
corrs2 = np.zeros([18, 8, 100, 2])

for ly in range(8):
    for sub in range(18):
        corrs1[sub, ly] = np.array(h5py.File("../../eeg-dcnn/corrs/sharpening/ly" + str((ly+1)*2) + "_sub" + str(sub+1) + ".h5", "r")["corrs"])
        corrs2[sub, ly] = np.array(h5py.File("../../eeg-dcnn/corrs/random_sharpening/ly" + str((ly+1)*2) + "_sub" + str(sub+1) + ".h5", "r")["corrs"])

rs1 = corrs1[:, :, :, 0]
rs2 = corrs2[:, :, :, 0]

for ly in range(8):
    for sub in range(18):
        for t in range(100):

            if t<=1:
                rs1[sub, ly, t] = np.average(rs1[sub, ly, :t+3])
                rs2[sub, ly, t] = np.average(rs2[sub, ly, :t+3])
            if t>1 and t<(100-2):
                rs1[sub, ly, t] = np.average(rs1[sub, ly, t-2:t+3])
                rs2[sub, ly, t] = np.average(rs2[sub, ly, t-2:t+3])
            if t>=(100-2):
                rs1[sub, ly, t] = np.average(rs1[sub, ly, t-2:])
                rs2[sub, ly, t] = np.average(rs2[sub, ly, t-2:])

ts = np.zeros([8, 100])
ps = np.zeros([8, 100])
for ly in range(8):
    for t in range(100):
        ts[ly, t], ps[ly, t] = ttest_rel(rs1[:, ly, t], rs2[:, ly, t])
        if ps[ly, t] < 0.05 and ts[ly, t] > 0:
            ps[ly, t] = 1
        else:
            ps[ly, t] = -1

print(ts)

ps10 = np.zeros([8, 1000])
ts10 = np.zeros([8, 1000])

for ly in range(8):
    for i in range(100):
        ps10[ly, i*10:i*10+10] = ps[ly, i]
        ts10[ly, i*10:i*10+10] = ts[ly, i]

newp = np.zeros([10, 1002])
newp[1:9, 1:1001] = ps10

newt = np.zeros([10, 1002])
newt[1:9, 1:1001] = ts10

x = np.linspace(-0.501, 1.501, 1002)
y = np.linspace(-0-0.04, 0.68, 10)
X, Y = np.meshgrid(x, y)
plt.contour(X, Y, newp, levels=[0], colors="r", linewidths=3, linestyles="dashed")

fig = plt.gcf()
fig.set_size_inches(16, 5.4)
#plt.imshow(newt, extent=(-0.51, 1.51, 0-0.04, 0.64+0.04), clim=(-1.5, 6.5), origin="lower", cmap="YlOrRd")
plt.imshow(ts, extent=(-0.5, 1.5, 0, 0.64), clim=(-0.5, 6.5), origin="lower", cmap="YlOrRd")
cb = plt.colorbar(ticks=[0, 6])
cb.ax.tick_params(labelsize=18)
font = {'size': 20}
cb.set_label("t", fontdict=font)

plt.tick_params(labelsize=18)
x = [0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6]
y = ["Layer2", "Layer4", "Layer6", "Layer8", "Layer10", "Layer12", "Layer14", "Layer16"]
plt.yticks(x, y, fontsize=18)
plt.xlim(-0.2, 1.5)
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("DCNN Layer", fontsize=20)
figpath = "results/hotmap/sharpening_valid_similarity.jpg"
plt.savefig(figpath, dpi=2000)
plt.show()

corrs1 = np.zeros([18, 8, 100, 2])
corrs2 = np.zeros([18, 8, 100, 2])

for ly in range(8):
    for sub in range(18):
        corrs1[sub, ly] = np.array(h5py.File("../../eeg-dcnn/corrs/sharpening_3faces/ly" + str((ly+1)*2) + "_sub" + str(sub+1) + ".h5", "r")["corrs"])
        corrs2[sub, ly] = np.array(h5py.File("../../eeg-dcnn/corrs/random_sharpening_3faces/ly" + str((ly+1)*2) + "_sub" + str(sub+1) + ".h5", "r")["corrs"])

rs1 = corrs1[:, :, :, 0]
rs2 = corrs2[:, :, :, 0]

for ly in range(8):
    for sub in range(18):
        for t in range(100):

            if t<=1:
                rs1[sub, ly, t] = np.average(rs1[sub, ly, :t+3])
                rs2[sub, ly, t] = np.average(rs2[sub, ly, :t+3])
            if t>1 and t<(100-2):
                rs1[sub, ly, t] = np.average(rs1[sub, ly, t-2:t+3])
                rs2[sub, ly, t] = np.average(rs2[sub, ly, t-2:t+3])
            if t>=(100-2):
                rs1[sub, ly, t] = np.average(rs1[sub, ly, t-2:])
                rs2[sub, ly, t] = np.average(rs2[sub, ly, t-2:])

ts = np.zeros([8, 100])
ps = np.zeros([8, 100])
for ly in range(8):
    for t in range(100):
        ts[ly, t], ps[ly, t] = ttest_rel(rs1[:, ly, t], rs2[:, ly, t])
        if ps[ly, t] < 0.05 and ts[ly, t] > 0:
            ps[ly, t] = 1
        else:
            ps[ly, t] = -1

print(ts)

ps10 = np.zeros([8, 1000])
ts10 = np.zeros([8, 1000])

for ly in range(8):
    for i in range(100):
        ps10[ly, i*10:i*10+10] = ps[ly, i]
        ts10[ly, i*10:i*10+10] = ts[ly, i]

newp = np.zeros([10, 1002])
newp[1:9, 1:1001] = ps10

newt = np.zeros([10, 1002])
newt[1:9, 1:1001] = ts10

x = np.linspace(-0.501, 1.501, 1002)
y = np.linspace(-0-0.04, 0.68, 10)
X, Y = np.meshgrid(x, y)
plt.contour(X, Y, newp, levels=[0], colors="r", linewidths=3, linestyles="dashed")

fig = plt.gcf()
fig.set_size_inches(16, 5.4)
#plt.imshow(newt, extent=(-0.51, 1.51, 0-0.04, 0.64+0.04), clim=(-1.5, 6.5), origin="lower", cmap="YlOrRd")
plt.imshow(ts, extent=(-0.5, 1.5, 0, 0.64), clim=(-0.5, 6.5), origin="lower", cmap="YlOrRd")
cb = plt.colorbar(ticks=[0, 6])
cb.ax.tick_params(labelsize=18)
font = {'size': 20}
cb.set_label("t", fontdict=font)

plt.tick_params(labelsize=18)
x = [0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6]
y = ["Layer2", "Layer4", "Layer6", "Layer8", "Layer10", "Layer12", "Layer14", "Layer16"]
plt.yticks(x, y, fontsize=18)
plt.xlim(-0.2, 1.5)
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("DCNN Layer", fontsize=20)
figpath = "results/hotmap/sharpening_3faces_valid_similarity.jpg"
plt.savefig(figpath, dpi=2000)
plt.show()