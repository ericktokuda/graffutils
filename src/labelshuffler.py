#!/usr/bin/env python3
"""Shuffle labels and do a z hypothesis test
"""

import os
import numpy as np
from logging import debug, info
from os.path import join as pjoin
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class LabelShuffler:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 1:
            info('Please provide the (1)labels and (2)the ids paths. Aborting...')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args))
            return

        labelspath = args[0]
        idspath = args[1]

        df = pd.read_csv(labelspath, index_col='id')

        nrealizations = 100
        nrows = df.shape[0]
        labels_all, counts_all = np.unique(df.label, return_counts=True)
        nlabels = len(labels_all)
        
        ########################################################## ids1
        ids = [int(x) for x in open(idspath).read().strip().split('\n')]

        # assuming there is at least of each class in each
        count_reg_orig = np.zeros(nlabels)
        aux1, aux2 = np.unique(df[df.index.isin(ids)].label, return_counts=True)
        for i, j in enumerate(aux1):
            count_reg_orig[j-1] = aux2[i]


        counts_shuffled = np.zeros((nrealizations, nlabels))

        x = np.zeros((nrealizations, nlabels))
        for i in range(nrealizations):
            newlabels = df.label.copy()
            np.random.shuffle(newlabels)
            df['label'] = newlabels
            labels_reg = df[df.index.isin(ids)].label
            labels_, counts_ = np.unique(labels_reg, return_counts=True)

            for j, l in enumerate(labels_):
                counts_shuffled[i, l-1] = counts_[j]

        
        df['regionid'] = np.zeros(nrows, dtype=int)
        df['regionid'].iloc[ids] = 1
        print(np.unique(df.regionid, return_counts=True))

        print('Orig count:\t{}\nShuff mu:\t{}\nShuff std \t{}'.format(count_reg_orig.astype(float),
                                                   np.mean(counts_shuffled, axis=0),
                                                   np.std(counts_shuffled, axis=0)))

        df.to_csv(pjoin(outdir, 'results.csv'))

