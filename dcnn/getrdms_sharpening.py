from neurora.rdm_cal import bhvRDM
import numpy as np
import h5py
from sklearn.decomposition import PCA

for i in range(8):
    features = np.array(h5py.File("features/ly" + str((i+1)*2) + ".h5", "r")["activations"])
    print(features.shape)
    n1 = features.shape[1]
    pca = PCA(n_components=0.95)
    pca.fit(features)
    f1 = pca.transform(features)
    n = f1.shape[1]
    f2 = np.zeros([450, n])
    f3 = np.zeros([450, n])
    for j in range(450):
        fj = np.array(features[j].copy())
        index = np.argsort(np.abs(fj))
        n0 = 0
        for k in range(n1):
            if fj[k] == 0:
                n0 = n0 + 1
        fj[index[n0:int(n0+(n1-n0)*0.5)]] = 0
        f2[j] = pca.transform(np.reshape(fj, [1, n1]))[0]
        fj = np.array(features[j].copy())
        fj[index[n0:int(n0+(n1-n0)*0.05)]] = 0
        f3[j] = pca.transform(np.reshape(fj, [1, n1]))[0]
    print(f1[5, :30])
    print(f2[5, :30])
    print(f3[5, :30])
    f = np.zeros([1350, n])
    f[:150] = f1[:150]
    f[150:300] = f2[:150]
    f[300:450] = f3[:150]
    f[450:600] = f1[150:300]
    f[600:750] = f2[150:300]
    f[750:900] = f3[150:300]
    f[900:1050] = f1[300:]
    f[1050:1200] = f2[300:]
    f[1200:1350] = f3[300:]
    print(f[:10, :5])
    print(f[150:160, :5])
    f = np.reshape(f, [1350, n, 1])
    rdm = bhvRDM(f)
    f = h5py.File("rdms_sharpening/ly" + str((i+1)*2) + ".h5", "w")
    f.create_dataset("rdm", data=rdm)
    f.close()