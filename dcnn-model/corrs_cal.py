import numpy as np
import h5py
from neurora.rdm_corr import rdm_correlation_spearman

lys = ["ly2", "ly4", "ly6", "ly8", "ly10", "ly12", "ly14", "ly16"]

dcnn_rdms = np.zeros([8, 450, 450])

index = 0
for ly in lys:
    dcnn_rdms[index] = np.array(h5py.File("../dcnn/rdms_afterPCA/" + ly + ".h5", "r")["rdm"])
    index = index + 1

models = ["familiar", "unfamiliar", "familiar-unfamiliar", "face", "contour", "face-contour"]

model_rdms = np.zeros([6, 450, 450])

index = 0
for model in models:
    model_rdms[index] = np.array(h5py.File("../models/dcnn/modelrdm/" + model + "_model.h5", "r")["rdm"])
    index = index + 1

corrs = np.zeros([6, 8])
ps = np.zeros([6, 8])
for i in range(6):
    for j in range(8):
        corrs[i, j], ps[i, j] = rdm_correlation_spearman(model_rdms[i], dcnn_rdms[j])
        print(corrs[i, j], ps[i, j])

f_rlts = h5py.File("corrs/corrs.h5", "w")
f_rlts.create_dataset('corrs', data=corrs)
f_rlts.create_dataset('ps', data=ps)
f_rlts.close()

dcnn_rdms = np.zeros([8, 450, 450])

index = 0
for ly in lys:
    dcnn_rdms[index] = np.array(h5py.File("../dcnn/rdms_random_afterPCA/" + ly + ".h5", "r")["rdm"])
    index = index + 1

models = ["familiar", "unfamiliar", "familiar-unfamiliar", "face", "contour", "face-contour"]

model_rdms = np.zeros([6, 450, 450])

index = 0
for model in models:
    model_rdms[index] = np.array(h5py.File("../models/dcnn/modelrdm/" + model + "_model.h5", "r")["rdm"])
    index = index + 1

corrs = np.zeros([6, 8])
ps = np.zeros([6, 8])
for i in range(6):
    for j in range(8):
        corrs[i, j], ps[i, j] = rdm_correlation_spearman(model_rdms[i], dcnn_rdms[j])
        print(corrs[i, j], ps[i, j])

f_rlts = h5py.File("corrs/random_corrs.h5", "w")
f_rlts.create_dataset('corrs', data=corrs)
f_rlts.create_dataset('ps', data=ps)
f_rlts.close()