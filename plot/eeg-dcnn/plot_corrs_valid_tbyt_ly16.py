import numpy as np
import h5py
from neurora.rsa_plot import plot_tbyt_diff_decoding_acc

corrs1 = np.zeros([18, 100, 2])

for sub in range(18):
    corrs1[sub] = np.array(h5py.File("../../eeg-dcnn/corrs/fatigue/ly16_sub" + str(sub + 1) + ".h5", "r")["corrs"]) \
                 - np.array(h5py.File("../../eeg-dcnn/corrs/random_fatigue/ly16_sub" + str(sub + 1) + ".h5", "r")["corrs"])

corrs2 = np.zeros([18, 100, 2])

for sub in range(18):
    corrs2[sub] = np.array(h5py.File("../../eeg-dcnn/corrs/sharpening/ly16_sub" + str(sub + 1) + ".h5", "r")["corrs"]) \
                 - np.array(h5py.File("../../eeg-dcnn/corrs/random_sharpening/ly16_sub" + str(sub + 1) + ".h5", "r")["corrs"])

plot_tbyt_diff_decoding_acc(corrs1[:, :, 0], corrs2[:, :, 0], start_time=-0.5, end_time=1.5, time_interval=0.02,
                            chance=0, p=0.05, cbpt=True, clusterp=0.05, stats_time=[0, 1.5], color1='olive',
                            color2='yellowgreen', label1='Fatigue', label2='Sharpening', xlim=[-0.25, 1.5],
                            ylim=[-0.15, 0.35], xlabel='Time (s)', ylabel='Valid Similarity', ticksize=14, fontsize=24,
                            markersize=4, legend_fontsize=16, title=None, title_fontsize=16, avgshow=False)