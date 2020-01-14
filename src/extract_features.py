#!/usr/bin/env python3
"""Extract features from images in a given directory using img2vec
"""

import argparse
import logging
import os
from os.path import join as pjoin
from logging import debug, info
import time
import pickle as pkl
import random

from img2vec_pytorch import Img2Vec
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imdir', required=True, help='Images directory')
    parser.add_argument('--outdir', default='/tmp', help='Output directory')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle files')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    img2vec = Img2Vec(cuda=True, model='resnet-18')

    imgs = sorted(os.listdir(args.imdir))
    if args.shuffle: random.shuffle(imgs)

    if not os.path.exists(args.outdir):
        info('Creating directory {}'.format(args.outdir))
        os.makedirs(args.outdir)

    for f in imgs:
        if not f.endswith('.jpg'): continue
        outpath = pjoin(args.outdir, f.replace('.jpg', '.pkl'))
        if os.path.exists(outpath): continue
        filepath = pjoin(args.imdir, f)

        img = Image.open(filepath)
        features = img2vec.get_vec(img)
        fh = open(outpath, 'wb')
        pkl.dump(features, fh)
        fh.close()

def run_one_experiment_given_list(l):
    run_experiment(*l)

if __name__ == "__main__":
    main()


