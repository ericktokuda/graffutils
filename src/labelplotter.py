#!/usr/bin/env python3
"""Bar plots of the graffiti occurrences
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

class LabelPlotter:
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
        cluareaspath = args[1]
        df = pd.read_csv(labelspath, index_col='id')
        totalrows = len(df)

        clusters = np.unique(df.cluster)
        clusters_str = ['C{}'.format(cl) for cl in clusters]
        nclusters = len(clusters)
        labels = np.unique(df.label)
        nlabels = len(labels)
        # labels_str = [str(l) for l in labels]
        plotsize = 5

        alpha = 0.7
        palette = np.array([
            [27.0,158,119],
            [217,95,2],
            [117,112,179],
            [231,41,138],
            [102,166,30],
            [230,171,2],
        ])
        palette /= 255.0
        colours = np.zeros((palette.shape[0], 4), dtype=float)
        colours[:, :3] = palette
        colours[:, -1] = alpha

        # Plot by type
        fig, ax = plt.subplots(1, nlabels,
                               figsize=(nlabels*plotsize, 1*plotsize),
                               squeeze=False)

        clustersums = np.zeros((nclusters, nlabels))
        for i, cl in enumerate(clusters):
            aux = df[df.cluster == cl]
            for j, l in enumerate(labels):
                clustersums[i, j] = len(aux[aux.label == l])

        labelsmap = {1: 'A', 2: 'B', 3: 'C'}
        for i, l in enumerate(labels):
            data = df[df.label == l]
            ys = np.zeros(nclusters)
            for k, cl in enumerate(clusters):
                ys[k] = len(data[data.cluster == cl]) / np.sum(clustersums[k, :])
                # print(cl, l, d)
            ax[0, i].bar(clusters_str, ys, color=colours)
            ax[0, i].axhline(y=len(df[df.label == l])/totalrows, linestyle='--')
            ax[0, i].set_title('Type {}'.format(labelsmap[l]))
            ax[0, i].set_ylabel('Ratio of Type {} inside community'.\
                                format(labelsmap[l]))
        # print(df.describe())
        fig.suptitle('Ratio of types of graffiti inside each cluster')
        plt.savefig(pjoin(outdir, 'count_per_type.pdf'))
        fig.clear()

        # Plot overall count
        fig, ax = plt.subplots(1, 1, figsize=(plotsize, plotsize),
                               squeeze=False)

        counts = np.zeros(nclusters, dtype=int)
        countsnorm = np.zeros(nclusters)
        areas = pd.read_csv(cluareaspath)

        for i, cl in enumerate(clusters):
            data = df[df.cluster == cl]
            counts[i] = len(data)
            points = data[['x', 'y']].values

            # from scipy.spatial import ConvexHull, convex_hull_plot_2d
            # hull = ConvexHull(points)
            # area = hull.volume
            countsnorm[i] = counts[i] / areas.iloc[i]
            # print(i, counts[i], countsrel[i])

        # ax[0, 0].bar(clusters_str, count)
        cumsum = 0.0
        axs = []
        for i, cl in enumerate(clusters):
            axs.append(ax[0, 0].barh(0, counts[i], 0.5, left=cumsum,
                                     label=clusters_str[i],
                                     color=colours[i]))
            cumsum += counts[i]
        # search all of the bar segments and annotate
        for j in range(len(axs)):
            for i, patch in enumerate(axs[j].get_children()):
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = patch.get_height() + bl[1]
                ax[0, 0].text(x,y, '{}'.format(counts[j]), ha='center')

        ax[0, 0].set_ylim(0, 3)
        ax[0, 0].legend()
        fig.patch.set_visible(False)
        ax[0, 0].axis('off')
        plt.savefig(pjoin(outdir, 'counts.pdf'))

        fig, ax = plt.subplots(1, 1, figsize=(2*plotsize, plotsize),
                               squeeze=False)
        yfactor = 1
        ax[0, 0].bar(clusters_str, countsnorm / yfactor, color=colours)
        ax[0, 0].set_ylabel('Normalized count of graffitis')

        plt.savefig(pjoin(outdir, 'countsnormalized.pdf'))
