import h5py
import numpy as np
from neurora.rsa_plot import plot_rdm

rdms = np.array(h5py.File("../../eeg/acc-ctrdms/avg_tbt-acc-rdms.h5", "r")["rdm"])

rdm = rdms[80]

plot_rdm(rdm, cmap="rainbow", percentile=True)