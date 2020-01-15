#!/usr/bin/env python3
"""Generate dendogram

python src/generate_dendogram.py  --rrspath ~/results/graffiti/rrs_200.txt --featpath ~/results/graffiti/features_resnet18_sample1000.csv
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rrspath', required=True, help='Output from Luciano s method')
    parser.add_argument('--featpath', required=True, help='Original features path')
    parser.add_argument('--outdir', default='/tmp', help='Output dir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)


    rrs = pd.read_csv(args.rrspath, sep=' ', header=None,
                      names=['featid','x1','x2', 'rel'])
    featuresdf = pd.read_csv(args.featpath)
    features = featuresdf.values[:, 1:]

    nrows, ncols = rrs.shape
    nmax = 5 # n maximum values
    nrandom = nmax

    inds_max = np.argsort(rrs.rel.values)[-nmax:]
    features_max = features[:, inds_max]
    # values_max = rrs.iloc[inds_max].rel.values

    inds_random = np.random.choice(range(nrows), nrandom)
    features_random = features[:, inds_random]

    meths = ['single', 'ward', 'centroid']

    second_approch(features_max)

def first_approach(features):
    res = 5
    fig, ax = plt.subplots(3, 3, figsize=(3*res, 3*res))

    for i, data in enumerate([features, features_random, features_max]):
        for j, meth in enumerate(meths):
            generate_dendogram(data, meth, ax[i, j])

    for ax_, col in zip(ax[0], meths):
        ax_.set_title(col)

    for ax_, row in zip(ax[:, 0], ['All', 'Random', 'Lucianomax5']):
        ax_.set_ylabel(row + '  ', rotation=0, size='large')

    plotpath = pjoin(args.outdir, 'dendogram.pdf')
    plt.savefig(plotpath)


def second_approch(X):
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.datasets import load_iris
    from sklearn.cluster import AgglomerativeClustering

    def plot_dendrogram(model, distancethresh):
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        dendrogram(linkage_matrix, color_threshold=distancethresh, truncate_mode='level', p=0)

    distancethresh = 15
    model = AgglomerativeClustering(distance_threshold=distancethresh, n_clusters=None)

    model = model.fit(X)
    print(model.labels_)
    input(model.n_clusters_)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, distancethresh)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

def generate_dendogram(data, meth, ax):
    """Generate dendogram using the given @ax

    Args:
    data(np.ndarray): matrix with samples as rows and features as columns
    meth(str): method to be used by scipy linkage call
    ax(plt.axis): matplotlib axis
    """

    l = linkage(data, meth)
    k = fcluster(l, t=1.15, criterion='inconsistent')
    ret = dendrogram(l, ax=ax, no_labels=True)

if __name__ == "__main__":
    main()

