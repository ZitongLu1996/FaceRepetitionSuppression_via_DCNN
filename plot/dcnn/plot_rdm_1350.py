import h5py
import numpy as np
from neurora.rsa_plot import plot_rdm

rdm = np.array(h5py.File("../../dcnn/rdms_sharpening/ly16.h5", "r")["rdm"])
rdm_avg = np.zeros([9, 9])
for i in range(9):
    for j in range(9):
        rdm_avg[i, j] = np.average(rdm[i*150:i*150+150, j*150:j*150+150])
for i in range(9):
    rdm_avg[i, i] = 0
plot_rdm(rdm, cmap="rainbow", percentile=True)
plot_rdm(rdm_avg, cmap="rainbow", percentile=True)