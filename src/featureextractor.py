#!/usr/bin/env python3
"""Feature extraction class
"""

import argparse
import logging
import time
from os.path import join as pjoin
from logging import debug, info
import numpy as np
import os
from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle as pkl

class FeatureExtractor:
    def __init__(self):
        pass

    def run(self, args, outdir):
        if len(args) < 1:
            info('Please provide (1)imgs dir as argument')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args[0]))
            return

        imdir = args[0]

        imgs = sorted(os.listdir(imdir))
        np.random.shuffle(imgs)
        img2vec = Img2Vec(cuda=True, model='resnet-18')

        for f in imgs:
            if not f.endswith('.jpg'): continue
            outpath = pjoin(outdir, f.replace('.jpg', '.pkl'))
            if os.path.exists(outpath): continue
            filepath = pjoin(imdir, f)

            img = Image.open(filepath)
            features = img2vec.get_vec(img)
            fh = open(outpath, 'wb')
            pkl.dump(features, fh)
            fh.close()

