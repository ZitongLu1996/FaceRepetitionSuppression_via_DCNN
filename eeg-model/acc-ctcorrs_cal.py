import h5py
import numpy as np
from neurora.ctcorr_cal import ctsim_ctrdms_cal

models = ["familiar", "unfamiliar", "familiar-unfamiliar", "face", "contour", "face-contour", "condition"]

for sub in range(18):

    eegctrdms = np.array(h5py.File("../eeg/acc-ctrdms/sub" + str(sub+1) + ".h5", "r")["ctrdms"])
    print(eegctrdms.shape)

    f = h5py.File("acc-ctcorrs/sub" + str(sub+1) + ".h5", "w")

    for model in models:
        print(sub, model)
        modelrdm = np.array(h5py.File("../models/eeg/modelrdm/" + model + "_model.h5", "r")["rdm"])
        ctcorrs = ctsim_ctrdms_cal(eegctrdms, modelrdm, method='spearman')
        f.create_dataset(model, data=ctcorrs)

    f.close()