import numpy as np
import h5py
from neurora.rsa_plot import plot_ct_decoding_acc

corrs_fatigue = np.zeros([18, 100, 100, 2])

for sub in range(18):
    corrs_fatigue[sub] = np.array(h5py.File("../../eeg-dcnn/ctcorrs/fatigue/ly16_sub" + str(sub + 1) + ".h5", "r")["ctcorrs"]) \
                 - np.array(h5py.File("../../eeg-dcnn/ctcorrs/random_fatigue/ly16_sub" + str(sub + 1) + ".h5", "r")["ctcorrs"])

plot_ct_decoding_acc(corrs_fatigue[:, :, :, 0], start_timex=-0.5, end_timex=1.5, start_timey=-0.5, end_timey=1.5,
                     time_intervalx=0.02, time_intervaly=0.02, chance=0, p=0.05, cbpt=True, clusterp=0.05,
                     stats_timex=[0, 1.5], stats_timey=[0, 1.5], xlim=[-0.25, 1.5], ylim=[-0.25, 1.5],
                     clim=[-0.01, 0.15], xlabel='Time (s)',  ylabel='Time (s)', clabel='Valid Similarity',
                     figsize=[6.4, 4.8], cmap="summer", ticksize=12, fontsize=16, title=None, title_fontsize=16)

corrs_sharpening = np.zeros([18, 100, 100, 2])

for sub in range(18):
    corrs_sharpening[sub] = np.array(h5py.File("../../eeg-dcnn/ctcorrs/sharpening/ly16_sub" + str(sub + 1) + ".h5", "r")["ctcorrs"]) \
                 - np.array(h5py.File("../../eeg-dcnn/ctcorrs/random_sharpening/ly16_sub" + str(sub + 1) + ".h5", "r")["ctcorrs"])

plot_ct_decoding_acc(corrs_sharpening[:, :, :, 0], start_timex=-0.5, end_timex=1.5, start_timey=-0.5, end_timey=1.5,
                     time_intervalx=0.02, time_intervaly=0.02, chance=0, p=0.05, cbpt=True, clusterp=0.05,
                     stats_timex=[0, 1.5], stats_timey=[0, 1.5], xlim=[-0.25, 1.5], ylim=[-0.25, 1.5],
                     clim=[-0.01, 0.15], xlabel='Time (s)',  ylabel='Time (s)', clabel='Valid Similarity',
                     figsize=[6.4, 4.8], cmap="summer", ticksize=12, fontsize=16, title=None, title_fontsize=16)