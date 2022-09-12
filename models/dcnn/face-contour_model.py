import numpy as np
import h5py

rdm = np.ones([450, 450], dtype=np.float)

rdm[:300, :300] = 0
rdm[300:, 300:] = 0

for i in range(450):
    rdm[i, i] = 0

f = h5py.File("modelrdm/face-contour_model.h5", "w")
f.create_dataset("rdm", data=rdm)
f.close()

from neurora.rsa_plot import plot_rdm

plot_rdm(rdm)