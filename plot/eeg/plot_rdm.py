import h5py
import numpy as np
from neurora.rsa_plot import plot_rdm

rdms = np.array(h5py.File("../../eeg/acc-ctrdms/avg_tbt-acc-rdms.h5", "r")["rdm"])

rdm = rdms[62]

conditions = ["FN", "FI", "FD", "UN", "UI", "UD", "SN", "SI", "SD"]

plot_rdm(rdm, conditions=conditions, con_fontsize=16, cmap="plasma", lim=[0.49, 0.7])