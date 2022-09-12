import numpy as np
import h5py

conditions = ["fn", "fe", "fl", "un", "ue", "ul", "sn", "se", "sl"]

for sub in range(18):
    ctrdms = np.zeros([100, 100, 9, 9])
    for con1 in range(9):
        for con2 in range(9):
            if con1 < con2:
                accs = np.array(h5py.File("decoding_acc/" + conditions[con1] + "-" + conditions[con2] + "_sub" + str(sub+1) + ".h5", "r")["acc"])
                ctrdms[:, :, con1, con2] = accs
                ctrdms[:, :, con2, con1] = accs
    f = h5py.File("acc-ctrdms/sub" +str(sub+1) + ".h5", "w")
    f.create_dataset("ctrdms", data=ctrdms)
    f.close()