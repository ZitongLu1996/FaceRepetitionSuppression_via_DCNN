from neurora.rdm_cal import bhvRDM
import numpy as np
import h5py
from sklearn.decomposition import PCA

for i in range(8):
    features = np.array(h5py.File("features_random/ly" + str((i+1)*2) + ".h5", "r")["activations"])
    print(features.shape)
    pca = PCA(n_components=0.95)
    f = pca.fit_transform(features)
    n = f.shape[1]
    print(n)
    f = np.reshape(f, [450, n, 1])
    rdm = bhvRDM(f)
    f = h5py.File("rdms_random_afterPCA/ly" + str((i+1)*2) + ".h5", "w")
    f.create_dataset("rdm", data=rdm)
    f.close()