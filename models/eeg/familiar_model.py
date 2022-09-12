import numpy as np
import h5py

rdm = np.ones([9, 9], dtype=np.float)

rdm[:3, :3] = 0

for i in range(9):
    rdm[i, i] = 0

f = h5py.File("modelrdm/familiar_model.h5", "w")
f.create_dataset("rdm", data=rdm)
f.close()

from neurora.rsa_plot import plot_rdm

plot_rdm(rdm)