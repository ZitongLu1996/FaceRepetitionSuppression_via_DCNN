import matplotlib.pyplot as plt
import numpy as np
import h5py
from neurora.rdm_corr import rdm_correlation_pearson

corrs = np.zeros([8, 8, 2])

rdms = np.zeros([8, 450, 450])
rdms_random = np.zeros([8, 450, 450])

for i in range(8):
    rdms[i] = np.array(h5py.File("../../dcnn/rdms_afterPCA/ly" + str((i+1)*2) + ".h5", "r")["rdm"])
    rdms_random[i] = np.array(h5py.File("../../dcnn/rdms_random_afterPCA/ly" + str((i+1)*2) + ".h5", "r")["rdm"])

for i in range(8):
    for j in range(8):
        corrs[i, j] = rdm_correlation_pearson(rdms[i], rdms_random[j])

print(corrs[:, :, 1])