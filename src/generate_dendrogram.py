#!/usr/bin/env python3
"""Generate dendrogram

rm /tmp/clusters/ -rf && python src/generate_dendrogram.py  --rrspath ~/results/graffiti/rrs_200.txt --featpath ~/results/graffiti/features_resnet18_sample1000.csv --imdir ~/results/graffiti/20200101-deeplab/20200109-sp20180511_crops/crop/20180511-gsv_spcity --outdir /tmp/hieclust_analysis
"""

import argparse
import logging
import os
from os.path import join as pjoin
from logging import debug, info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
import random
from subprocess import Popen

def get_clusters(nclusters, data, meth, imgids, outdir):
    # Clustering
    model = AgglomerativeClustering(linkage=meth, n_clusters=nclusters)
    model = model.fit(data)
    l = model.labels_

    for i in range(nclusters):
        listpath = pjoin(outdir, '{}-{}clusters-{}.lst'.format(meth, nclusters,
                                                               chr(i+97)))
        inds = np.where(l == i)
        # imgs = imgids[inds]

        with open(listpath, 'w') as f:
            for item in imgids[inds]:
                f.write("%s.jpg\n" % item)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rrspath', required=True, help='Output from Luciano s method')
    parser.add_argument('--featpath', required=True, help='Original features path')
    parser.add_argument('--imdir', required=True, help='Images dir')
    parser.add_argument('--randomseed', default=0, help='Random seed')
    parser.add_argument('--outdir', default='/tmp', help='Output dir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    np.random.seed(args.randomseed)
    random.seed(args.randomseed)

    rrs = pd.read_csv(args.rrspath, sep=' ', header=None,
                      names=['featid','x1','x2', 'rel'])
    featuresdf = pd.read_csv(args.featpath)
    features = featuresdf.values[:, 1:]
    imgids = featuresdf.values[:, 0]

    nrows, ncols = rrs.shape
    nmax = 5 # n maximum values
    nrandom = nmax

    inds_max = np.argsort(rrs.rel.values)[-nmax:]
    features_max = features[:, inds_max]
    # values_max = rrs.iloc[inds_max].rel.values

    inds_random = np.random.choice(range(nrows), nrandom)
    features_random = features[:, inds_random]

    meths = ['single', 'ward', 'average']
    dataslices = ['All', 'Random', 'Lucianomax5']

    res = 5
    fig, ax = plt.subplots(3, 3, figsize=(3*res, 3*res))

    for i, data in enumerate([features, features_random, features_max]):
        for j, meth in enumerate(meths):
            generate_dendrogram(data, meth, ax[i, j])

    for ax_, col in zip(ax[0], meths):
        ax_.set_title(col)

    for ax_, row in zip(ax[:, 0], dataslices):
        ax_.set_ylabel(row + '  ', rotation=0, size='large')

    plotpath = pjoin(args.outdir, 'dendrogram.pdf')
    plt.savefig(plotpath)

    ########################################################## Analyze clusters
    clustersdir = pjoin(args.outdir, 'clusters')
    if os.path.exists(clustersdir):
        '{} exists. Not generating cluster lists'.format(clustersdir)
        return
    else:
        os.mkdir(clustersdir)

    for i, data in enumerate([features, features_random, features_max]):
        for j, meth in enumerate(meths):
            get_clusters(2, data, meth, imgids, clustersdir)
            get_clusters(4, data, meth, imgids, clustersdir)

    ########################################################## Copy images
    nimgs = 30
    files = os.listdir(clustersdir)
    for f in files:
        if os.path.isdir(f): continue
        dirpath = pjoin(clustersdir, os.path.splitext(f)[0])
        os.mkdir(dirpath)
        fh = open(pjoin(clustersdir, f))
        files = fh.read().splitlines()
        random.shuffle(files)
        for imgname in files[:nimgs]:
            cmd = 'cp {} {}'.format(pjoin(args.imdir, imgname), dirpath)
            pid = Popen(cmd, shell=True)
            print(pid)
##########################################################
def generate_dendrogram(X, meth, ax):
    distancethresh = 15
    model = AgglomerativeClustering(distance_threshold=distancethresh,
                                    linkage=meth, n_clusters=None)

    model = model.fit(X)

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

    dendrogram(linkage_matrix, color_threshold=distancethresh, ax=ax,
               no_labels=True)

if __name__ == "__main__":
    main()

