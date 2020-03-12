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
from matplotlib.ticker import FormatStrFormatter

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

        alpha = 0.6
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

        coloursrev = []
        for i in range(len(palette)):
            coloursrev.append(colours[len(palette) - 1 - i, :])

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

            barplot = ax[0, i].barh(list(reversed(clusters_str)), list(reversed(ys)),
                                    color=coloursrev)

            ax[0, i].axvline(x=len(df[df.label == l])/totalrows, linestyle='--')
            ax[0, i].text(len(df[df.label == l])/totalrows + 0.05,
                          -0.7, 'ref', ha='center', va='bottom',
                          rotation=0, color='royalblue',
                          fontsize='large')
            reftick = len(df[df.label == l])/totalrows
            ax[0, i].set_xlim(0, 1)
            ax[0, i].set_title('Ratio of Type {} within communities'.\
                                format(r"$\bf{" + str(labelsmap[l]) + "}$"),
                               size='large', pad=30)
            ax[0, i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[0, i].set_xticks([0, .25, 0.5, .75, 1.0])
            ax[0, i].spines['top'].set_color('gray')
            ax[0, i].xaxis.set_ticks_position('top')
            ax[0, i].tick_params(axis='x', which='both', length=0, colors='gray',
                                 labelsize='large')
            ax[0, i].tick_params(axis='y', which='both', labelsize='large')
            ax[0, i].xaxis.grid(True, alpha=0.4)
            ax[0, i].set_axisbelow(True)

            def autolabel(rects, ys):
                for idx, rect in enumerate(rects):
                    height = rect.get_height()
                    # print(rect.get_x(), height)
                    ax[0, i].text(rect.get_width()-0.05,
                                  rect.get_y() + rect.get_height()/2.-0.18,
                                  '{:.2f}'.format(ys[idx]), color='white',
                                  ha='center', va='bottom', rotation=0,
                                  fontsize='large')

            autolabel(barplot, list(reversed(ys)))
            # ax[0, i].axis("off")
            for spine in ax[0, i].spines.values():
                spine.set_edgecolor('dimgray')
            ax[0, i].spines['bottom'].set_visible(False)
            ax[0, i].spines['right'].set_visible(False)
            ax[0, i].spines['left'].set_visible(False)

        # plt.box(False)
        # fig.suptitle('Ratio of graffiti types inside each cluster', size='x-large')
        plt.tight_layout(pad=5)
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

            countsnorm[i] = counts[i] / areas.iloc[i]

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
        ax[0, 0].set_xlabel('Community')
        for spine in ax[0, i].spines.values():
            spine.set_edgecolor('dimgray')
        ax[0, i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax[0, i].spines['top'].set_visible(False)
        ax[0, i].spines['right'].set_visible(False)
        ax[0, i].yaxis.grid(True, alpha=0.4)
        ax[0, i].set_axisbelow(True)
        # ax[0, i].spines['left'].set_visible(False)

        plt.savefig(pjoin(outdir, 'countsnormalized.pdf'))
