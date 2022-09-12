import numpy as np
import h5py
from neurora.rsa_plot import plot_tbytsim_withstats

nsubs = 18

models = ['face', 'contour', 'face-contour', 'familiar', 'unfamiliar', 'familiar-unfamiliar', 'condition']

for model in models:
    rs = np.zeros([nsubs, 100, 100])

    index = 0
    for sub in range(nsubs):
        subctcorrs = np.array(h5py.File("../../eeg-model/acc-ctcorrs/sub" + str(index+1) +".h5", "r")[model])
        rs[index] = subctcorrs[:, :, 0]
        index = index + 1

    tyt_rs = np.zeros([nsubs, 100])
    for i in range(100):
        tyt_rs[:, i] = rs[:, i, i]

    print(model)
    plot_tbytsim_withstats(tyt_rs, start_time=-0.5, end_time=1.5, time_interval=0.02, smooth=True, p=0.05, cbpt=True,
                           clusterp=0.05, stats_time=[0, 1.5], color='green', xlim=[-0.5, 1.5], ylim=[-0.15, 0.55],
                           xlabel='Time (s)', ylabel='Representational Similarity')