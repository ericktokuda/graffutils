#!/usr/bin/env python3
""" Features clustering and export to pickle
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from skfuzzy.cluster import cmeans

from src import utils
from src.utils import info

##########################################################
def cluster(csvpath, algo, outdir):
    info(inspect.stack()[0][3] + '()')
    if not os.path.exists(outdir): os.mkdir(outdir)

    df = pd.read_csv(csvpath)
    matrix_ = df.values[:,1:]
    all = algo == 'all'

    if all or (algo == 'tsne'):
        cluster_tsne(matrix_, outdir)
    # if all or (algo == 'multicoretsne'):
        # cluster_multicoretsne(matrix_, outdir)
    if all or (algo == 'dbscan'):
        cluster_dbscan(matrix_, outdir)
    if all or (algo == 'kmeans'):
        cluster_kmeans(matrix_, outdir)
    if all or (algo == 'fuzzy'):
        cluster_fuzzy(matrix_, outdir)

##########################################################
def cluster_tsne(matrix_, outdir):
    info(inspect.stack()[0][3] + '()')
    for perpl in [20, 40, 60, 100]:
        t0 = time.time()
        tsne = TSNE(n_components=2, random_state=0, perplexity=perpl)
        t = tsne.fit_transform(matrix_)
        h5path = pjoin(outdir, 'tsne{}.h5'.format(perpl))
        utils.dump_to_hdf5(t, h5path, type=float)
        info('perplexity:{} elapsed time:{}'.format(perpl, time.time() - t0))

##########################################################
def cluster_multicoretsne(matrix_, outdir):
    from MulticoreTSNE import MulticoreTSNE as TSNE
    info(inspect.stack()[0][3] + '()')
    for perpl in [20, 40, 60]:
       t0 = time.time()
       tsne = TSNE(n_jobs=6, n_components=2, random_state=0,
               perplexity=perpl)
       t = tsne.fit_transform(matrix_)
       h5path = pjoin(outdir, 'multicoretsne{}.h5'.format(perpl))
       utils.dump_to_hdf5(t, h5path, type=float)
       info('perplexity:{} elapsed time:{}'.format(perpl, time.time() - t0))

##########################################################
def cluster_dbscan(matrix_, outdir):
    info(inspect.stack()[0][3] + '()')
    for eps in [0.1, 0.5, 1.0, 3.0, 5.0]:
       t0 = time.time()
       clustering = DBSCAN(eps=eps, min_samples=2).fit(matrix_)
       h5path = pjoin(outdir, 'dbscan{:.1f}.h5'.format(eps))
       utils.dump_to_hdf5(clustering.labels_, h5path, type=int)
       info('epsilon:{} elapsed time:{}'.format(eps, time.time() - t0))

##########################################################
def cluster_kmeans(matrix_, outdir):
    info(inspect.stack()[0][3] + '()')
    for ncomp in [2, 3, 4, 5]:
       t0 = time.time()
       clustering = KMeans(n_clusters=ncomp, random_state=0).fit(matrix_)
       h5path = pjoin(outdir, 'kmeans{}.h5'.format(ncomp))
       utils.dump_to_hdf5(clustering.labels_, h5path, type=int)
       info('ncomp:{} elapsed time:{}'.format(ncomp, time.time() - t0))

##########################################################
def cluster_fuzzy(matrix_, outdir):
    info(inspect.stack()[0][3] + '()')
    for ncomp in [2, 3, 4, 5]:
       t0 = time.time()
       clustering = KMeans(n_clusters=ncomp, random_state=0).fit(matrix_)
       # res = cmeans(matrix_, ncomp, m=2, error=0.005,
               # maxiter=10000, init=None)
       h5path = pjoin(outdir, 'fuzzy{}.h5'.format(ncomp))
       utils.dump_to_hdf5(clustering.labels_, h5path, type=int)
       info('ncomp:{} elapsed time:{}'.format(ncomp, time.time() - t0))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csvpath', required=True, help='Csv path')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    cluster(args.csvpath, 'all', args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

