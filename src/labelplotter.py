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
        df = pd.read_csv(labelspath, index_col='id')
        totalrows = len(df)

        clusters = np.unique(df.cluster)
        clusters_str = ['C{}'.format(cl) for cl in clusters]
        nclusters = len(clusters)
        labels = np.unique(df.label)
        nlabels = len(labels)
        # labels_str = [str(l) for l in labels]
        plotsize = 5

        fig, ax = plt.subplots(nlabels, 1,
                               figsize=(1*plotsize, nlabels*plotsize),
                               squeeze=False)

        clustersums = np.zeros((nclusters, nlabels))
        for i, cl in enumerate(clusters):
            aux = df[df.cluster == cl]
            for j, l in enumerate(labels):
                clustersums[i, j] = len(aux[aux.label == l])

        for i, l in enumerate(labels):
            data = df[df.label == l]
            ys = np.zeros(nclusters)
            for k, cl in enumerate(clusters):
                ys[k] = len(data[data.cluster == cl]) / np.sum(clustersums[k, :])
                # print(cl, l, d)
            ax[i, 0].bar(clusters_str, ys)
            ax[i, 0].axhline(y=len(df[df.label == l])/totalrows, linestyle='--')
            ax[i, 0].set_title('Type {}'.format(l))
        # print(df.describe())
        fig.suptitle('Ratio of types of graffiti inside each cluster')
        plt.savefig('/tmp/out.pdf')
