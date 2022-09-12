import numpy as np
import h5py
from neurora.corr_cal_by_rdm import rdms_corr

for ly in range(8):
    vggrdm1350 = np.array(h5py.File("../dcnn/rdms_fatigue_3faces/ly"+str((ly+1)*2)+".h5", "r")["rdm"])
    vggrdm = np.zeros([9, 9])
    for i in range(9):
        for j in range(9):
            vggrdm[i, j] = np.average(vggrdm1350[i * 150:i * 150 + 150, j * 150:j * 150 + 150])
    for i in range(9):
        vggrdm[i, i] = 0

    for sub in range(18):
        eegctrdms = np.array(h5py.File("../eeg/acc-ctrdms/sub" + str(sub + 1) + ".h5", "r")["ctrdms"])

        eegrdms = np.zeros([100, 9, 9])

        for t in range(100):
            eegrdms[t] = eegctrdms[t, t]

        corrs = rdms_corr(vggrdm, eegrdms)

        f = h5py.File("corrs/fatigue_3faces/ly" + str((ly+1)*2) + "_sub" + str(sub+1) +".h5", "w")
        f.create_dataset("corrs", data=corrs)
        f.close()

        print(str((ly+1)*2), str(sub+1))