#!/usr/bin/env python3
"""Analyze labels generated by mannual classification of the crops
"""

import argparse
import time
from os.path import join as pjoin
import os
import inspect

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from scipy.spatial import cKDTree
import scipy.stats as stats
import igraph

# from src import utils
from src.utils import info

##########################################################
def plot_cluster_labels(clulabelspath, cluareaspath, outdir):
    """Plot cluster labels.
    It expects a df with the points and corresponding cluster, objlabel
    and also a list of the areas of each cluster.
    """

    info(inspect.stack()[0][3] + '()')

    df = pd.read_csv(clulabelspath, index_col='id')
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

    counts = np.zeros(nclusters, dtype=int)
    countsnorm = np.zeros(nclusters)
    areas = pd.read_csv(cluareaspath)

    for i, cl in enumerate(clusters):
        data = df[df.cluster == cl]
        counts[i] = len(data)
        points = data[['x', 'y']].values

        countsnorm[i] = counts[i] / areas.iloc[i]

    fig, ax = plt.subplots(1, 1, figsize=(2*plotsize, plotsize),
                           squeeze=False)
    yfactor = 1
    ax[0, 0].bar(clusters_str, countsnorm / yfactor, color=colours)
    ax[0, 0].set_ylabel('Normalized count of graffitis')
    ax[0, 0].set_xlabel('Community')
    i = 0
    for spine in ax[0, i].spines.values():
        spine.set_edgecolor('dimgray')
    ax[0, i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[0, i].spines['top'].set_visible(False)
    ax[0, i].spines['right'].set_visible(False)
    ax[0, i].yaxis.grid(True, alpha=0.4)
    ax[0, i].set_axisbelow(True)
    # ax[0, i].spines['left'].set_visible(False)

    plt.savefig(pjoin(outdir, 'countsnormalized.pdf'))

##########################################################
def count_labels_per_region(df, clusters, labels, cluids):
    """Count number of labels per region
    """
    nlabels = len(labels)
    nclusters = len(clusters)

    counts = np.ones((nclusters, nlabels), dtype=float)
    for i in range(nclusters):
        labels_reg, counts_reg = np.unique(df[df.index.isin(cluids[i])].label,
                                           return_counts=True)
        for j in range(nlabels):
            lab = labels[j]
            if not lab in labels_reg: continue
            ind = np.where(labels_reg == lab)
            counts[i, j] = counts_reg[ind]
    return counts

##########################################################
def count_shuffled_labels_per_region(dforig, clusters, labels, cluids, nrealizations):
    """Shuffle and count for @nrealizations times
    Returns an array indexed by the nrealization, regionid, objtype
    """
    info(inspect.stack()[0][3] + '()')

    counts_perm = np.ones((nrealizations, len(clusters), len(labels)),
                             dtype=float) * 999

    for i in range(nrealizations):
        df = dforig.copy()
        newlabels = df.label.copy()
        np.random.shuffle(newlabels)
        df['label'] = newlabels
        counts_perm[i, :, :] = count_labels_per_region(df, clusters, labels, cluids)
    return counts_perm

#############################################################
def shuffle_labels(labelspath, outdir):
    """Shuffle labels from @labelspath and compute metrics
    """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(labelspath, index_col='id')

    nrealizations = 10
    labels = np.unique(df.label)
    clusters = np.unique(df.cluster)
    nlabels = len(labels)
    nclusters = len(clusters)
    info('nrealizations:{}, nclusters:{}'.format(nrealizations, nclusters))

    cluids = {}
    for i in range(nclusters):
        cluids[i] = df[df.cluster == clusters[i]].index

    counts_orig = count_labels_per_region(df, clusters, labels, cluids)
    counts_perm = count_shuffled_labels_per_region(df, clusters, labels,
            cluids, nrealizations)

    plot_shuffle_distrib_and_orig(counts_orig, counts_perm, nclusters,
            nlabels, outdir)

##########################################################
def plot_shuffle_distrib_and_orig(counts_orig, counts_perm, nclusters,
        nlabels, outdir):
    """Plot shuffle counts distribs and original counts
    """
    info(inspect.stack()[0][3] + '()')
    nperregion = np.sum(counts_orig, axis=1)
    plotsize = 5
    fig, ax = plt.subplots(1, nclusters,
                           figsize=(nclusters*plotsize, 1*plotsize),
                           squeeze=False)

    palette = np.array([
        [127,201,127, 255],
        [190,174,212, 255],
        [253,192,134, 255],
        [255,255,153, 255],
    ])
    palette = palette / 255

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
    plt.savefig(pjoin(outdir, 'counts_shuffled_norm.pdf'))

#############################################################
def summarize_annotations(annotdir, outdir):
    """Summarize @annotdir csv annotations in .txt format and output
    summary to @outdir
    """
    info(inspect.stack()[0][3] + '()')
    if not os.path.exists(outdir): os.mkdir(outdir)

    files = sorted(os.listdir(annotdir))

    labels = '1 2 3'.split(' ')
    info('Using labels:{}'.format(labels))

    cols = 'img,x,y,label'.split(',')
    data = []
    for f in files:
        if not f.endswith('.txt'): continue
        filepath = pjoin(annotdir, f)
        _, y, x, heading = os.path.split(filepath)[-1].replace('.txt', '').split('_')
        labels_ = open(filepath).read().strip().split(',')

        for l in labels_: # each label in the file correspond to a new row
            img = f.replace('.txt', '.jpg')
            data.append([img, x, y, l])

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(pjoin(outdir, 'labels.csv'), index_label='id',)

    for l in labels: # open
        listpath = os.path.join(outdir, 'label_{}.lst'.format(l))
        df[df.label == l].to_csv(listpath, columns=['img'], index=False)

    cmd = '# export PREV=${{PWD}} && for I in {}; '\
        'do mkdir {}/label_${{I}} -p && '\
        'cd {} && '\
        'xargs --arg-file {}/label_${{I}}.lst '\
        'cp --target-directory="{}/label_${{I}}/"; done && '\
        'cd ${{PREV}}'. \
        format(' '.join(labels), outdir, '<IMDIR>', outdir, outdir)
    info('# If you want to copy the images, replace <IMDIR> and run:')
    info(cmd)

##########################################################
def parse_infomap_output(graphmlpath, infomapout, labelspath, outdir):
    """Find enclosing community given by @infomapout of each node in @graphml
    """
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'clusters.csv')

    g = igraph.Graph.Read(graphmlpath)

    cludf = pd.read_csv(infomapout, sep=' ', skiprows=[0, 1],
                     names=['id', 'cluster','flow']) # load graph clusters
    cludf = cludf.sort_values(by=['id'], inplace=False)

    objsdf = pd.read_csv(labelspath, index_col='id') # load obj labels
    pd.set_option("display.precision", 8)

    coords_objs = np.zeros((len(objsdf), 2))
    for idx, row in objsdf.iterrows():
        coords_objs[idx, 0] = row.x
        coords_objs[idx, 1] = row.y

    coords_nodes = np.array([g.vs['x'], g.vs['y']]).T

    kdtree = cKDTree(coords_nodes)
    dists, inds = kdtree.query(coords_objs)
    objsdf['cluster'] = np.array(cludf.cluster.tolist())[inds]
    objsdf.to_csv(outpath)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('--annotdir', required=True, help='Annotations directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    # summarize_annotations(args.annotdir, args.outdir)
    graphmlpath = '/home/dufresne/temp/20200202-types/20200221-citysp.graphml'
    infomapout = '/home/dufresne/temp/20200202-types/20200222-citysp_infomap.clu'
    clulabelspath = '/home/dufresne/temp/20200202-types/20200209-cityspold_8003_labels_clu.csv'
    cluareaspath = '/home/dufresne/temp/20200202-types/20200222-citysp_infomap_areas.csv'

    # parse_infomap_output(graphmlpath, infomapout, labelspath, args.outdir)
    # shuffle_labels(labelspath, args.outdir)
    # plot_cluster_labels(clulabelspath, cluareaspath, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

