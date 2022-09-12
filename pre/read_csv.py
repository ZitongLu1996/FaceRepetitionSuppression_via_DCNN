# -*- coding: utf-8

"""
@File       :   read_csv.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import csv

ntrials = np.zeros([18, 9], dtype=np.int)

for i in range(18):
    filepath = '/Users/zitonglu/Desktop/face_perception/eeg_dataset/sub-'+str(i+2).zfill(3)+'/eeg/events.csv'
    with open(filepath)as f:
        f_csv = csv.reader(_.replace('\x00', '') for _ in f)
        index = np.zeros([9], dtype=np.int)
        for row in f_csv:
            #print(row)
            row = np.array(row)
            #print(row.shape)
            #print(row[3])
            strs = ['famous_new', 'famous_second_early', 'famous_second_late', 'unfamiliar_new', 'unfamiliar_second_early', 'unfamiliar_second_late', 'scrambled_new', 'scrambled_second_early', 'scrambled_second_late']
            for j in range(9):
                if row[3] == strs[j]:
                    index[j] = index[j] + 1
        print(index)
    ntrials[i] = index

np.savetxt("ntrials_info.txt", ntrials)