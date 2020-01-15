#!/usr/bin/env python3
"""Cluster data
"""

import argparse
import logging
import os
from os.path import join as pjoin
from logging import debug, info
import pickle as pkl

import pandas as pd
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import skfuzzy
from skfuzzy.cluster import cmeans

def cluster_tsne(matrix_, outdir):
    info('Clustering using T-SNE')
    for perpl in [20, 40, 60, 100]:
        t0 = time.time()
        tsne = TSNE(n_components=2, random_state=0, perplexity=perpl)
        t = tsne.fit_transform(matrix_)
        elapsed = time.time() - t0
        pklpath = pjoin(outdir, 'tsne{}.pkl'.format(perpl))
        pkl.dump(t, open(pklpath, 'wb'))
        info('perplexity:{} elapsed time:{}'.format(perpl, time.time() - t0))

def cluster_multicoretsne(matrix_, outdir):
    from MulticoreTSNE import MulticoreTSNE as TSNE
    info('Clustering using T-SNE')
    for perpl in [20, 40, 60]:
        t0 = time.time()
        tsne = TSNE(n_jobs=6, n_components=2, random_state=0,
                    perplexity=perpl)
        t = tsne.fit_transform(matrix_)
        elapsed = time.time() - t0
        pklpath = pjoin(outdir, 'multicoretsne{}.pkl'.format(perpl))
        pkl.dump(t, open(pklpath, 'wb'))
        info('perplexity:{} elapsed time:{}'.format(perpl, time.time() - t0))

def cluster_dbscan(matrix_, outdir):
    info('Clustering using DBSCAN')
    for eps in [0.1, 0.5, 1.0, 3.0, 5.0]:
        t0 = time.time()
        clustering = DBSCAN(eps=eps, min_samples=2).fit(matrix_)
        pklpath = pjoin(outdir, 'dbscan{:.1f}.pkl'.format(eps))
        pkl.dump(clustering, open(pklpath, 'wb'))
        info('epsilon:{} elapsed time:{}'.format(eps, time.time() - t0))

def cluster_kmeans(matrix_, outdir):
    info('Clustering using K-means')
    for ncomp in [2, 3, 4, 5]:
        t0 = time.time()
        clustering = KMeans(n_clusters=ncomp, random_state=0).fit(matrix_)
        pklpath = pjoin(outdir, 'kmeans{}.pkl'.format(ncomp))
        pkl.dump(clustering, open(pklpath, 'wb'))
        info('ncomp:{} elapsed time:{}'.format(ncomp, time.time() - t0))

def cluster_fuzzy(matrix_, outdir):
    info('Clustering using fuzzy c-means')
    for ncomp in [2, 3, 4, 5]:
        t0 = time.time()
        clustering = KMeans(n_clusters=ncomp, random_state=0).fit(matrix_)
        res = cmeans(matrix_, ncomp, m=2, error=0.005,
                     maxiter=10000, init=None)
        pklpath = pjoin(outdir, 'fuzzy{}.pkl'.format(ncomp))
        pkl.dump(clustering, open(pklpath, 'wb'))
        info('ncomp:{} elapsed time:{}'.format(ncomp, time.time() - t0))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='csv file')
    parser.add_argument('--outdir', default='/tmp', help='output dir')
    parser.add_argument('--algo', default='all', help='clustering algorithm')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    df = pd.read_csv(args.csv)
    matrix_ = df.values[:,1:]

    if args.algo == 'all': all = True
    else: all = False

    if all or args.algo.lower() == 'tsne':
        cluster_tsne(matrix_, args.outdir)
    if all or args.algo.lower() == 'multicoretsne':
        cluster_multicoretsne(matrix_, args.outdir)
    if all or args.algo.lower() == 'dbscan':
        cluster_dbscan(matrix_, args.outdir)
    if all or args.algo.lower() == 'kmeans':
        cluster_kmeans(matrix_, args.outdir)
    if all or args.algo.lower() == 'fuzzy':
        cluster_fuzzy(matrix_, args.outdir)

if __name__ == "__main__":
    main()

