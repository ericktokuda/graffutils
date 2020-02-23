#!/usr/bin/env python3
"""Shuffle labels from csv

python src/shuffle_labels.py  --labelspath /home/frodo/temp/20200209-sample_8003_annot_labels/labels.csv --idspath /home/frodo/temp/region1_ids.txt
"""

import argparse
import logging
import time
from os.path import join as pjoin
from logging import debug, info
import pandas as pd
import numpy as np

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labelspath', required=True, help='labels path')
    parser.add_argument('--idspath', required=True, help='ids path')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    np.random.seed(0)
    np.set_printoptions(precision=2, formatter={'float':lambda x: '{:05.2f}'.format(x)})
    df = pd.read_csv(args.labelspath, index_col='id')

    nrealizations = 100
    nrows = df.shape[0]
    labels_all, counts_all = np.unique(df.label, return_counts=True)
    nlabels = len(labels_all)
    
    idspath1 = '/home/frodo/results/graffiti/20200215-region1_ids.txt'
    idspath2 = '/home/frodo/results/graffiti/20200215-region2_ids.txt'

    ########################################################## ids1
    # ids = [int(x) for x in open(args.idspath).read().strip().split('\n')]
    ids = [int(x) for x in open(idspath1).read().strip().split('\n')]

    # assuming there is at least of each class in each
    count_reg_orig = np.zeros(nlabels)
    aux1, aux2 = np.unique(df[df.index.isin(ids)].label, return_counts=True)
    for i, j in enumerate(aux1):
        count_reg_orig[j-1] = aux2[i]


    counts_shuffled = np.zeros((nrealizations, nlabels))

    x = np.zeros((nrealizations, nlabels))
    for i in range(nrealizations):
        newlabels = df.label.copy()
        np.random.shuffle(newlabels)
        df['label'] = newlabels
        labels_reg = df[df.index.isin(ids)].label
        labels_, counts_ = np.unique(labels_reg, return_counts=True)

        for j, l in enumerate(labels_):
            counts_shuffled[i, l-1] = counts_[j]

    
    df['regionid'] = np.zeros(nrows, dtype=int)
    df['regionid'].iloc[ids] = 1
    # df.loc[df.isin(ids), 'regionid'] = 1
    print(np.unique(df.regionid, return_counts=True))

    print('Orig count:\t{}\nShuff mu:\t{}\nShuff std \t{}'.format(count_reg_orig.astype(float),
                                               np.mean(counts_shuffled, axis=0),
                                               np.std(counts_shuffled, axis=0)))

    ########################################################## ids1
    # ids = [int(x) for x in open(args.idspath).read().strip().split('\n')]
    ids = [int(x) for x in open(idspath2).read().strip().split('\n')]

    # assuming there is at least of each class in each
    count_reg_orig = np.zeros(nlabels)
    aux1, aux2 = np.unique(df[df.index.isin(ids)].label, return_counts=True)
    for i, j in enumerate(aux1):
        count_reg_orig[j-1] = aux2[i]


    counts_shuffled = np.zeros((nrealizations, nlabels))

    x = np.zeros((nrealizations, nlabels))
    for i in range(nrealizations):
        newlabels = df.label.copy()
        np.random.shuffle(newlabels)
        df['label'] = newlabels
        labels_reg = df[df.index.isin(ids)].label
        labels_, counts_ = np.unique(labels_reg, return_counts=True)

        for j, l in enumerate(labels_):
            counts_shuffled[i, l-1] = counts_[j]

    
    df['regionid'].iloc[ids] = 2
    print(np.unique(df.regionid, return_counts=True))



    print('Orig count:\t{}\nShuff mu:\t{}\nShuff std \t{}'.format(count_reg_orig.astype(float),
                                               np.mean(counts_shuffled, axis=0),
                                               np.std(counts_shuffled, axis=0)))

    ##########################################################
    df.to_csv('/tmp/foo.csv')

    info('Elapsed time:{}'.format(time.time()-t0))
if __name__ == "__main__":
    main()

