import numpy as np
import h5py
from neurora.rsa_plot import plot_ct_decoding_acc

nsubs = 18

models = ['face', 'contour', 'face-contour', 'familiar', 'unfamiliar', 'familiar-unfamiliar', 'condition']

for model in models:
    rs = np.zeros([nsubs, 100, 100])

    index = 0
    for sub in range(nsubs):
        subctcorrs = np.array(h5py.File("../../eeg-model/acc-ctcorrs/sub" + str(index+1) +".h5", "r")[model])
        rs[index] = subctcorrs[:, :, 0]
        index = index + 1

    print(model)
    plot_ct_decoding_acc(rs, start_timex=-0.5, end_timex=1.5, start_timey=-0.5, end_timey=1.5, time_intervalx=0.02,
                         time_intervaly=0.02, chance=0, p=0.05, cbpt=True, clusterp=0.05, stats_timex=[0, 1.5],
                         stats_timey=[0, 1.5], xlim=[-0.25, 1.5], ylim=[-0.25, 1.5], clim=[-0.01, 0.2],
                         xlabel='Time (s)',  ylabel='Time (s)', clabel='Representational Similarity',
                         figsize=[6.4, 4.8], cmap="YlGn", ticksize=12, fontsize=16, title=None, title_fontsize=16)