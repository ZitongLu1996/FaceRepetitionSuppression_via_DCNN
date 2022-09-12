import h5py
import numpy as np
from neurora.rsa_plot import plot_rdm

models = ["familiar", "unfamiliar", "familiar-unfamiliar", "face", "contour", "face-contour"]

for model in models:

    modelrdm = np.array(h5py.File("../../../modelrdms/dcnn/modelrdm/" + model + "_model.h5", "r")["rdm"])
    plot_rdm(modelrdm, cmap="YlGn")