import h5py
import numpy as np
from neurora.rsa_plot import plot_rdm

rdm = np.array(h5py.File("../../dcnn/rdms_random_afterPCA/ly4.h5", "r")["rdm"])
print(rdm.shape)
plot_rdm(rdm, cmap="rainbow", percentile=True)