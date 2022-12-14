import numpy as np
import h5py

rdm = np.ones([9, 9], dtype=np.float)

rdm = np.array([[0, 1, 0.2, 0, 1, 0.2, 0, 1, 0.2],
                [1, 0, 0.8, 1, 0, 0.8, 1, 0, 0.8],
                [0.2, 0.8, 0, 0.2, 0.8, 0, 0.2, 0.8, 0],
                [0, 1, 0.2, 0, 1, 0.2, 0, 1, 0.2],
                [1, 0, 0.8, 1, 0, 0.8, 1, 0, 0.8],
                [0.2, 0.8, 0, 0.2, 0.8, 0, 0.2, 0.8, 0],
                [0, 1, 0.2, 0, 1, 0.2, 0, 1, 0.2],
                [1, 0, 0.8, 1, 0, 0.8, 1, 0, 0.8],
                [0.2, 0.8, 0, 0.2, 0.8, 0, 0.2, 0.8, 0]])

"""rdm[:15, :15] = 0
rdm[15:21, 15:21] = 0
rdm[21:27, 21:27] = 0
rdm[27:42, 27:42] = 0
rdm[42:48, 42:48] = 0
rdm[48:54, 48:54] = 0
rdm[54:69, 54:69] = 0
rdm[69:75, 69:75] = 0
rdm[75:, 75:] = 0
rdm[:15, 15:21] = 0.5
rdm[:15, 21:27] = 0
rdm[15:21, :15] = 0.5
rdm[15:21, 21:27] = 0.5
rdm[21:27, :15] = 0
rdm[21:27, 15:21] = 0.5
rdm[27:42, 42:48] = 0.5
rdm[42:48, 27:42] = 0.5
rdm[27:42, 48:54] = 0
rdm[48:54, 27:42] = 0
rdm[42:48, 48:54] = 0.5
rdm[48:54, 42:48] = 0.5
rdm[54:69, 69:75] = 0.5
rdm[69:75, 54:69] = 0.5
rdm[54:69, 75:] = 0
rdm[75:, 54:69] = 0
rdm[69:75, 75:] = 0.5
rdm[75:, 69:75] = 0.5

for i in range(81):
    rdm[i, i] = 0"""

f = h5py.File("modelrdm/condition_model.h5", "w")
f.create_dataset("rdm", data=rdm)
f.close()
from neurora.rsa_plot import plot_rdm

plot_rdm(rdm)