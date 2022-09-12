import numpy as np
import h5py
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
acc = np.zeros([8, 3])
for i in range(8):
    features = np.array(h5py.File("features_random/ly" + str((i+1)*2) + ".h5", "r")["activations"])
    print(features.shape)
    pca = PCA(n_components=0.95)
    features = pca.fit_transform(features)
    print("PCA")
    f_features = features[:150]
    u_features = features[150:300]
    s_features = features[300:]
    labels = np.zeros([300])
    labels[150:] = 1
    fu_acc = np.zeros([100])
    fs_acc = np.zeros([100])
    us_acc = np.zeros([100])
    for k in range(100):
        state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((f_features, u_features), axis=0), labels, test_size=1 / 3, random_state=state)
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        fu_acc[k] = svm.score(x_test, y_test)
        print("familiar")
        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((f_features, s_features), axis=0), labels, test_size=1 / 3, random_state=state)
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        fs_acc[k] = svm.score(x_test, y_test)
        print("unfamiliar")
        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((u_features, s_features), axis=0), labels, test_size=1 / 3, random_state=state)
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        us_acc[k] = svm.score(x_test, y_test)
        print("scrambled")
    acc[i, 0] = np.average(fu_acc)
    acc[i, 1] = np.average(fs_acc)
    acc[i, 2] = np.average(us_acc)
    print(acc[i])
np.savetxt("classification_results/acc_random.txt", acc)