#!/usr/bin/env python3
"""Clustering
"""

import argparse
import logging
import time
import os
from os.path import join as pjoin
from logging import debug, info
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import skfuzzy
from skfuzzy.cluster import cmeans
import sys
import pickle as pkl


class Clustering:
    def __init__(self):
        pass

    def run(self, args, outdir):
        algos = ['all', 'tsne', 'dbscan', 'kmeans', 'fuzzy']
        if len(args) < 2:
            info('Please provide (1)method and (2)csvpath. Aborting...')
            return
        elif not args[0] in algos:
            info('Please provide choose one among {}. Aborting...'.format(algos))
            return
        elif not os.path.exists(args[1]):
            info('Please check if {} exists'.format(args[1]))
            return

        if args[0] == 'all':
            algo = algos
        else:
            algo = [args[0]]
        csvpath = args[1]

        df = pd.read_csv(csvpath)
        matrix_ = df.values[:,1:]


        if 'tsne' in algo:
            self.cluster_tsne(matrix_, outdir)
        if 'multicoretsne' in algo:
            self.cluster_multicoretsne(matrix_, outdir)
        if 'dbscan' in algo:
            self.cluster_dbscan(matrix_, outdir)
        if 'kmeans' in algo:
            self.cluster_kmeans(matrix_, outdir)
        if 'fuzzy' in algo:
            self.cluster_fuzzy(matrix_, outdir)

    def cluster_tsne(self, matrix_, outdir):
       info('Clustering using T-SNE')
       for perpl in [20, 40, 60, 100]:
           t0 = time.time()
           tsne = TSNE(n_components=2, random_state=0, perplexity=perpl)
           t = tsne.fit_transform(matrix_)
           elapsed = time.time() - t0
           pklpath = pjoin(outdir, 'tsne{}.pkl'.format(perpl))
           pkl.dump(t, open(pklpath, 'wb'))
           info('perplexity:{} elapsed time:{}'.format(perpl, time.time() - t0))

    def cluster_multicoretsne(self, matrix_, outdir):
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

    def cluster_dbscan(self, matrix_, outdir):
       info('Clustering using DBSCAN')
       for eps in [0.1, 0.5, 1.0, 3.0, 5.0]:
           t0 = time.time()
           clustering = DBSCAN(eps=eps, min_samples=2).fit(matrix_)
           pklpath = pjoin(outdir, 'dbscan{:.1f}.pkl'.format(eps))
           pkl.dump(clustering, open(pklpath, 'wb'))
           info('epsilon:{} elapsed time:{}'.format(eps, time.time() - t0))

    def cluster_kmeans(self, matrix_, outdir):
       info('Clustering using K-means')
       for ncomp in [2, 3, 4, 5]:
           t0 = time.time()
           clustering = KMeans(n_clusters=ncomp, random_state=0).fit(matrix_)
           pklpath = pjoin(outdir, 'kmeans{}.pkl'.format(ncomp))
           pkl.dump(clustering, open(pklpath, 'wb'))
           info('ncomp:{} elapsed time:{}'.format(ncomp, time.time() - t0))

    def cluster_fuzzy(self, matrix_, outdir):
       info('Clustering using fuzzy c-means')
       for ncomp in [2, 3, 4, 5]:
           t0 = time.time()
           clustering = KMeans(n_clusters=ncomp, random_state=0).fit(matrix_)
           res = cmeans(matrix_, ncomp, m=2, error=0.005,
                        maxiter=10000, init=None)
           pklpath = pjoin(outdir, 'fuzzy{}.pkl'.format(ncomp))
           pkl.dump(clustering, open(pklpath, 'wb'))
           info('ncomp:{} elapsed time:{}'.format(ncomp, time.time() - t0))

