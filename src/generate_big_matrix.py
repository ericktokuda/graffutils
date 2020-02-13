#!/usr/bin/env python3
"""Assemble all pickle in one single file
"""

import argparse
import logging
import os
from os.path import join as pjoin
from logging import debug, info
import pickle as pkl
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkldir', required=True, help='Pickles directory')
    parser.add_argument('--samplesize', default='all', help='Sample size.Defaults to all. ')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    random.seed(0)

    files = np.array(os.listdir(args.pkldir))


    fh = open('/tmp/out.csv', 'w')
    fh.write('id,lat,lon')
    # arr = [ 'feat{:03d}'.format(x) for x in range(512)]
    # fh.write(','.join(arr))
    fh.write('\n')

    inds = list(range(len(files)))
    if args.samplesize == 'all':
        files = sorted(files)
    else:
        print(type(args.samplesize))
        print((args.samplesize))
        random.shuffle(inds)
        inds = inds[:int(args.samplesize)]
        files = files[inds]

    for i, f in enumerate(files):
        pklpath = pjoin(args.pkldir, f)
        if not os.path.exists(pklpath): continue
        print('f:{}'.format(f))
        feat = pkl.load(open(pklpath, 'rb'))
        feat = [str(x) for x in feat]
        lat = (f.strip().split('_')[1])
        lon = (f.strip().split('_')[2])
        # fh.write('{:04d},{},{},{},'.format(inds[i], os.path.splitext(f)[0], lat, lon))
        fh.write('{:04d},{},{}'.format(inds[i], lat, lon))
        # fh.write(','.join(feat))
        fh.write('\n')

if __name__ == "__main__":
    main()

