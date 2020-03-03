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

        nrealizations = 500
        plotsize = 5
        nrows = df.shape[0]
        labels_uniq, counts_uniq = np.unique(df.label, return_counts=True)
        clusters_uniq, counts_uniq = np.unique(df.cluster, return_counts=True)
        nlabels = len(labels_uniq)
        nclusters = len(clusters_uniq)
        info('nrealizations:{}, nclusters:{}'.format(nrealizations, nclusters))

        counts_orig = np.ones((nclusters, nlabels), dtype=float)
        counts_perm = np.ones((nrealizations, nclusters, nlabels),
                                 dtype=float) * 999

        ids_reg = {}
        for i in range(nclusters):
            ids_reg[i] = df[df.cluster == clusters_uniq[i]].index

        ########################################################## orig
        for i in range(nclusters):
            labels_reg, counts_reg = np.unique(df[df.index.isin(ids_reg[i])].label,
                                               return_counts=True)
            for j in range(nlabels):
                lab = labels_uniq[j]
                if not lab in labels_reg: continue
                ind = np.where(labels_reg == lab)
                counts_orig[i, j] = counts_reg[ind]

        ########################################################## permutations
        for i in range(nrealizations):
            aux = df.label.copy()
            np.random.shuffle(aux)
            df['label'] = aux

            for j in range(nclusters):
                labels_reg, counts_reg = np.unique(df[df.index.isin(ids_reg[j])].label,
                                                   return_counts=True)
                for k in range(nlabels):
                    lab = labels_uniq[k]
                    if not lab in labels_reg: continue

                    ind = np.where(labels_reg == lab)
                    counts_perm[i, j, k] = counts_reg[ind]

        ########################################################## Plot mean/std

        labelsstr = ['type' + str(l) for l in np.arange(nlabels)]
        fig, ax = plt.subplots(1, nclusters,
                               figsize=(nclusters*plotsize, 1*plotsize),
                               squeeze=False, sharey='row')

        for j in range(nclusters):
            data = counts_perm[:, j, :]
            ax[0, j].errorbar(labelsstr, np.mean(data, axis=0), np.std(data, axis=0),
                              fmt='o', c='r')
            ax[0, j].scatter(labelsstr, counts_orig[j, :], marker='D',
                             c='b')
            ax[0, j].set_xlabel('Graffiti type')
            ax[0, j].set_ylabel('Count')
        
        fig.suptitle('Z-test for the counts of the three types of '\
                        'graffiti for each cluster')
        plt.savefig(pjoin(outdir, 'label_shuffling.png'))
        plt.clf()

        ########################################################## Plot distributions
        import scipy.stats as stats
        labelsstr = ['type' + str(l) for l in np.arange(nlabels)]
        fig, ax = plt.subplots(1, nclusters,
                               figsize=(nclusters*plotsize, 1*plotsize),
                               squeeze=False)
                               # squeeze=False, sharey='row')

        # print(counts_perm)
        # input()
        palette = np.array([
            [127,201,127, 255],
            [190,174,212, 255],
            [253,192,134, 255],
            [255,255,153, 255],
        ])
        palette = palette / 255
        epsilon = 0.01

        for j in range(nclusters):
            for k in range(nlabels):
                data = counts_perm[:, j, k]
                density = stats.gaussian_kde(data)
                # density = gaussian_kde(data)
                xs = np.linspace(0, 220, num=110)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                ax[0, j].plot(xs, density(xs), label=str(k), c=palette[k])

                count_ref = density(counts_orig[j, k])
                ax[0, j].scatter(counts_orig[j, k], count_ref, c=[palette[k]])
                if density(counts_orig[j, k]) > epsilon:
                    ax[0, j].quiver(counts_orig[j, k], 0.0, 0.0, count_ref,
                                    angles='xy', scale_units='xy', scale=1,
                                    linestyle='dashed', color=palette[k])

            ax[0, j].legend()
            ax[0, j].set_xlabel('Graffiti count')
        
        fig.suptitle('Distrubution of number of graffiti occurences '\
                        'per type (colours) and per cluster (plots)')

        plt.savefig(pjoin(outdir, 'label_shuffling_distribs.pdf'))

        ########################################################## Plot mean/std relative
        labelsstr = ['type' + str(l) for l in np.arange(nlabels)]
        fig, ax = plt.subplots(1, nclusters,
                               figsize=(nclusters*plotsize, 1*plotsize),
                               squeeze=False, sharey='row')

        nperregion = np.sum(counts_orig, axis=1)
        for j in range(nclusters):
            data = counts_perm[:, j, :] / nperregion[j]

            ax[0, j].errorbar(labelsstr, np.mean(data, axis=0), np.std(data, axis=0),
                              fmt='o', c='r')
            ax[0, j].scatter(labelsstr, counts_orig[j, :] / nperregion[j], marker='D',
                             c='b')
            ax[0, j].set_xlabel('Graffiti type')
            ax[0, j].set_ylabel('Count')
        
        fig.suptitle('Z-test for the counts of the three types of '\
                        'graffiti for each cluster')
        plt.savefig(pjoin(outdir, 'label_shuffling_relative.png'))
        plt.clf()

        ########################################################## Plot distributions relative
        import scipy.stats as stats
        labelsstr = ['type' + str(l) for l in np.arange(nlabels)]
        fig, ax = plt.subplots(1, nclusters,
                               figsize=(nclusters*plotsize, 1*plotsize),
                               squeeze=False)
                               # squeeze=False, sharey='row')

        # print(counts_perm)
        # input()
        palette = np.array([
            [127,201,127, 255],
            [190,174,212, 255],
            [253,192,134, 255],
            [255,255,153, 255],
        ])
        palette = palette / 255
        epsilon = 0.01

        for j in range(nclusters):
            for k in range(nlabels):
                data = counts_perm[:, j, k] / nperregion[j]
                density = stats.gaussian_kde(data)
                # density = gaussian_kde(data)
                xs = np.linspace(0, 1, num=100)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                ys = density(xs)
                ys /= np.sum(ys)
                ax[0, j].plot(xs, ys, label=str(k), c=palette[k])

                count_ref = counts_orig[j, k] / nperregion[j]
                ax[0, j].scatter(count_ref, 0, c=[palette[k]])
                plt.text(0.5, 0.9, 'samplesz:{:.0f}'.format(nperregion[j]),
                         horizontalalignment='center', verticalalignment='center',
                         fontsize='large', transform = ax[0, j].transAxes)


            ax[0, j].legend()
            ax[0, j].set_xlabel('Graffiti relative count')
        
        fig.suptitle('Distrubution of number of graffiti occurences '\
                        'per type (colours) and per cluster (plots)')

        print(nperregion)
        plt.savefig(pjoin(outdir, 'label_shuffling_distribs_relative.pdf'))

        ##########################################################
        df.to_csv(pjoin(outdir, 'results.csv'), index=False)


