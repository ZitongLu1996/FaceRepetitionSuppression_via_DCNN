import numpy as np
import h5py

from neurora.rsa_plot import plot_ct_decoding_acc
49
nsubs = 18

cons = ["fn-fe", "fn-fl", "fe-fl", "un-ue", "un-ul", "ue-ul", "sn-se", "sn-sl", "se-sl",
        "fn-un", "fn-sn", "un-sn", "fe-ue", "fe-se", "ue-se", "fl-ul", "fl-sl", "ul-sl"]

for con in cons:
    accs = np.zeros([nsubs, 100, 100])
    for sub in range(nsubs):
        subaccs = np.array(h5py.File("../../../eeg/decoding_acc/" + con + "_sub" + str(sub + 1) + ".h5", "r")["acc"])
        accs[sub] = subaccs

    plot_ct_decoding_acc(accs, start_timex=-0.5, end_timex=1.5, start_timey=-0.5, end_timey=1.5, time_intervalx=0.02,
                         time_intervaly=0.02, chance=0.5, p=0.01, cbpt=True, clusterp=0.05, stats_timex=[0, 1.5],
                         stats_timey=[0, 1.5], xlim=[-0.25, 1.5], ylim=[-0.25, 1.5], clim=[0.49, 0.6],
                         xlabel='Training Time (s)', ylabel='Test Time (s)', clabel='Decoding Accuracy',
                         figsize=[6.4, 4.8], cmap="inferno", ticksize=12, fontsize=16, title=None, title_fontsize=16)