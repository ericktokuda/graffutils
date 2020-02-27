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
import matplotlib
import matplotlib.pyplot as plt

class LabelShuffler:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 1:
            info('Please provide the (1)csv file with a field cluster and'\
                 ' a field label. Aborting...')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args))
            return

        labelspath = args[0]

        df = pd.read_csv(labelspath, index_col='id')

        nrealizations = 100
        plotsize = 5
        nrows = df.shape[0]
        labels_all, counts_all = np.unique(df.label, return_counts=True)
        nlabels = len(labels_all)
        nclusters = len(np.unique(df.cluster))
        info('nrealizations:{}, nclusters:{}'.format(nrealizations, nclusters))

        counts = np.ones((nclusters, nlabels), dtype=float)
        means = np.ones((nclusters, nlabels), dtype=float)
        stds = np.ones((nclusters, nlabels), dtype=float)

        ########################################################## ids
        for k, cl in enumerate(np.unique(df.cluster)):
            ids = df[df.cluster == cl].index

            # assuming there is at least of each class in each region
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


            counts[k, :] = count_reg_orig.astype(float)
            means[k, :] = np.mean(counts_shuffled, axis=0)
            stds[k, :] = np.std(counts_shuffled, axis=0)

            info('Orig count:\t{}\nShuff mu:\t{}\nShuff std \t{}'.format(count_reg_orig.astype(float),
                                                       np.mean(counts_shuffled, axis=0),
                                                       np.std(counts_shuffled, axis=0)))

        df = pd.DataFrame(index=np.arange(nclusters))
        for label in range(nlabels):
            countkey = 'count{}'.format(label)
            shufflemeankey = 'shuffle{}mean'.format(label)
            shufflestdkey = 'shuffle{}std'.format(label)

            df[countkey] = counts[:, label]
            df[shufflemeankey] = means[:, label]
            df[shufflestdkey] = stds[:, label]

        palette = ['r', 'g', 'b']

        # plot
        labelsstr = ['type' + str(l) for l in np.arange(nlabels)]
        fig, ax = plt.subplots(1, nclusters,
                               figsize=(nclusters*plotsize, 1*plotsize),
                               squeeze=False, sharey='row')

        for cl in range(nclusters):
            ax[0, cl].errorbar(labelsstr, means[cl, :], stds[cl, :],
                               fmt='o', c='r')
            ax[0, cl].scatter(labelsstr, counts[cl, :], marker='D',
                              c='b')
            ax[0, cl].set_xlabel('Graffiti type')
            ax[0, cl].set_ylabel('Count')
        
        fig.suptitle('Z-test for the counts of the three types of '\
                        'graffiti for each cluster')
        plt.savefig(pjoin(outdir, 'label_shuffling.png'))
        df.to_csv(pjoin(outdir, 'results.csv'), index=False)


