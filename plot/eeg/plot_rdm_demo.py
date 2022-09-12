import h5py
import numpy as np
from neurora.rsa_plot import plot_rdm

rdm = np.random.random([9, 9])
rdm = 0.495 + rdm/20
for i in range(9):
    for j in range(9):
        if i > j:
            rdm[i, j] = rdm[j, i]
for i in range(9):
    rdm[i, i] = 0

conditions = ["FN", "FE", "FL", "UN", "UE", "UL", "SN", "SE", "SL"]

plot_rdm(rdm, conditions=conditions, con_fontsize=16, cmap="plasma", lim=[0.49, 0.7])