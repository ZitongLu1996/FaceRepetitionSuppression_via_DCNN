import numpy as np
import h5py
from neurora.ctcorr_cal import ctsim_ctrdms_cal

vggrdm1350 = np.array(h5py.File("../dcnn/rdms_fatigue_3faces/ly16.h5", "r")["rdm"])
vggrdm = np.zeros([9, 9])
for i in range(9):
    for j in range(9):
        vggrdm[i, j] = np.average(vggrdm1350[i * 150:i * 150 + 150, j * 150:j * 150 + 150])
for i in range(9):
    vggrdm[i, i] = 0

for sub in range(18):
    eegctrdms = np.array(h5py.File("../eeg/acc-ctrdms/sub" + str(sub + 1) + ".h5", "r")["ctrdms"])

    corrs = ctsim_ctrdms_cal(eegctrdms, vggrdm)

    f = h5py.File("ctcorrs/fatigue_3faces/ly16_sub" + str(sub+1) +".h5", "w")
    f.create_dataset("ctcorrs", data=corrs)
    f.close()

    print(str(sub+1))