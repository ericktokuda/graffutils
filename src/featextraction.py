#!/usr/bin/env python3
"""Feature extraction 
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
from datetime import datetime
from PIL import Image

from img2vec_pytorch import Img2Vec
import src.utils2 as utils

HOME = os.getenv('HOME')
#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def extract_features_all(imdir, outdir):
    """Extract features from all images in @imdir and export to @outdir
    """
    info(inspect.stack()[0][3] + '()')

    files = sorted(os.listdir(imdir))
    np.random.shuffle(files)
    img2vec = Img2Vec(cuda=True, model='resnet-18')

    for f in files:
        if not f.endswith('.jpg'): continue
        outpath = pjoin(outdir, f.replace('.jpg', '.h5'))
        if os.path.exists(outpath): continue
        filepath = pjoin(imdir, f)

        img = Image.open(filepath)
        features = img2vec.get_vec(img)
        utils.dump_to_hdf5(features, outpath)

##########################################################
def format_features(ind, h5path):
    feat = utils.read_hdf5(h5path)
    filename = os.path.basename(h5path)
    f = os.path.splitext(filename)[0]
    featstr = [str(x) for x in feat]
    _, lat, lon, _, _ = f.strip().split('_')
    row = '{:04d},{},{},{},'.format(ind, f, lat, lon)
    row += ','.join(featstr) + '\n'
    return row

##########################################################
def concatenate_features_sample(featdir, samplesz, outpath):
    info(inspect.stack()[0][3] + '()')

    files = np.array(sorted(os.listdir(featdir)))

    fh = open(pjoin(outdir, 'features.csv'), 'w')
    fh.write('id,file,lat,lon,')
    arr = [ 'feat{:03d}'.format(x) for x in range(512)]
    fh.write(','.join(arr))
    fh.write('\n')

    inds = list(range(len(files)))
    if samplesz == -1:
        files = sorted(files)
    else:
        np.random.shuffle(inds)
        inds = inds[:int(samplesize)]
        files = files[inds]

    for i, f in enumerate(files):
        pklpath = pjoin(featdir, f)
        if not os.path.exists(pklpath): continue
        info('f:{}'.format(f))
        feat = pkl.load(open(pklpath, 'rb'))
        feat = [str(x) for x in feat]
        lat = (f.strip().split('_')[1])
        lon = (f.strip().split('_')[2])
        breakpoint()
        fh.write('{:04d},{},{},{},'.format(inds[i], os.path.splitext(f)[0], lat, lon))
        fh.write(','.join(feat))
        fh.write('\n')

##########################################################
def concatenate_features_all(featdir, featpath):
    """Concatenate features in @featdir into the csv file @featpath.
    We avoid constructing a pandas dataframe because it may consume a
    lot of memory.
    """
    info(inspect.stack()[0][3] + '()')
    if not os.path.exists(outdir): os.mkdir(outdir)
    info(inspect.stack()[0][3] + '()')

    files = np.array(sorted(os.listdir(featdir)))

    fh = open(featpath, 'w') 
    fh.write('id,file,lat,lon,')
    arr = [ 'feat{:03d}'.format(x) for x in range(512)]
    fh.write(','.join(arr))
    fh.write('\n')

    inds = list(range(len(files)))
    files = sorted(files)

    ind_h5 = 0
    for f in files:
        if not f.endswith('.h5'): continue
        info('f:{}'.format(f))
        h5path = pjoin(featdir, f)
        fh.write(format_features(ind_h5, h5path))
        ind_h5 += 1

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    imdir = pjoin(HOME, 'results/graffiti/20200202-types/20200109-sp20180511_crops/crop/20180511-gsv_spcity')
    featdir = args.outdir
    featpath = pjoin(args.outdir, 'features.csv')

    # extract_features_all(imdir, featdir)
    concatenate_features_all(featdir, featpath)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

