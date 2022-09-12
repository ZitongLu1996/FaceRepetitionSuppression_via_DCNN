import numpy as np
import scipy.io as sio
from neurora.decoding import ct_decoding_holdout, ct_decoding_kfold
import h5py

# 1, 7, 13
nsubs = 18
nts = 100
nchls = 70
tstep = 5
navgts = 5
navgtrials = 5
data_path = "../EEGdata/030hz"
conditions = ["fn", "fe", "fl", "un", "ue", "ul", "sn", "se", "sl"]
nconditions = len(conditions)

for con1 in range(9):
    for con2 in range(9):
        for sub in range(nsubs):
            if con1 < con2:
                print(conditions[con1], conditions[con2], sub)
                con1_rawdata = sio.loadmat(data_path + "/sub" + str(sub + 1) + "_" + conditions[con1] + ".mat")["data"]
                con2_rawdata = sio.loadmat(data_path + "/sub" + str(sub + 1) + "_" + conditions[con2] + ".mat")["data"]
                ntrials1 = np.shape(con1_rawdata)[2]
                ntrials2 = np.shape(con2_rawdata)[2]
                con1_data = np.zeros([nchls, 500, ntrials1])
                con2_data = np.zeros([nchls, 500, ntrials2])
                con1_data[:60] = con1_rawdata[:60]
                con1_data[60:] = con1_rawdata[64:]
                con2_data[:60] = con2_rawdata[:60]
                con2_data[60:] = con2_rawdata[64:]
                data = np.zeros([1, ntrials1+ntrials2, nchls, 500])
                data[0, :ntrials1] = np.transpose(con1_data, (2, 0, 1))
                data[0, ntrials1:] = np.transpose(con2_data, (2, 0, 1))
                labels = np.zeros([1, ntrials1+ntrials2])
                labels[0, ntrials1:] = 1

                acc = ct_decoding_holdout(data, labels, n=2, navg=5, time_opt='average', time_win=5, time_step=5,
                                        iter=10, test_size=1/3)[0]

                f = h5py.File("decoding_acc/" + conditions[con1] + "-" + conditions[con2] + "_sub" + str(sub + 1) +
                              ".h5", "w")
                f.create_dataset("acc", data=acc)
                f.close()