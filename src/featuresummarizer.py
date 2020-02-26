#!/usr/bin/env python3
"""Assemble all pickle in one single file
"""

from logging import debug, info
import pickle as pkl
import numpy as np
import os
from os.path import join as pjoin

class FeatureSummarizer:
    def __init__(self):
        pass

    def run(self, args, outdir):
        if len(args) < 2:
            info('Please provide the (1)pickes dir and (2)size of the sample as arguments')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exists'.format(args[0]))
            return

        pkldir = args[0]
        samplesize = args[1]
        np.random.seed(0)
        files = np.array(os.listdir(pkldir))

        fh = open(pjoin(outdir, 'out.csv'), 'w')
        fh.write('id,file,lat,lon,')
        arr = [ 'feat{:03d}'.format(x) for x in range(512)]
        fh.write(','.join(arr))
        fh.write('\n')

        inds = list(range(len(files)))
        if samplesize == 'all':
            files = sorted(files)
        else:
            np.random.shuffle(inds)
            inds = inds[:int(samplesize)]
            files = files[inds]

        print(files)

        for i, f in enumerate(files):
            pklpath = pjoin(pkldir, f)
            if not os.path.exists(pklpath): continue
            print('f:{}'.format(f))
            feat = pkl.load(open(pklpath, 'rb'))
            feat = [str(x) for x in feat]
            lat = (f.strip().split('_')[1])
            lon = (f.strip().split('_')[2])
            fh.write('{:04d},{},{},{},'.format(inds[i], os.path.splitext(f)[0], lat, lon))
            # fh.write('{:04d},{},{}'.format(inds[i], lat, lon))
            fh.write(','.join(feat))
            fh.write('\n')


