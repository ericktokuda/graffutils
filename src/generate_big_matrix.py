#!/usr/bin/env python3
"""Assemble all pickle in one single file
"""

import argparse
import logging
import os
from os.path import join as pjoin
from logging import debug, info
import pickle as pkl

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkldir', required=True, help='Pickles directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    files = sorted(os.listdir(args.pkldir))

    fh = open('/tmp/out.csv', 'w')
    fh.write('id,')
    arr = [ 'feat{:03d}'.format(x) for x in range(512)]
    fh.write(','.join(arr))
    fh.write('\n')

    for f in files:
        # info('file:{}'.format(f))
        pklpath = pjoin(args.pkldir, f)
        if not os.path.exists(pklpath): continue
        print(args.pkldir, f, pklpath)
        feat = pkl.load(open(pklpath, 'rb'))
        feat = [str(x) for x in feat]
        fh.write('{},'.format(os.path.splitext(f)[0]))
        fh.write(','.join(feat))
        fh.write('\n')

if __name__ == "__main__":
    main()

