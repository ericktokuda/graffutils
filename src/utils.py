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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import sys
import h5py
import inspect
from datetime import datetime

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def dump_to_hdf5(x, h5path, type=float):
    """Dump @x to @outdir/@filename
    """
    info(inspect.stack()[0][3] + '()')

    if os.path.exists(h5path): os.remove(h5path)
    fh = h5py.File(h5path, "w")
    dset = fh.create_dataset("data", data=x, dtype=type)
    fh.close()

##########################################################
def analyze_deeplab_log(logpath):
    """Parse deeplab log in an attempt to find the best iou
    It is a bit crypt here.
    """
    info(inspect.stack()[0][3] + '()')
    if not os.path.exists(outdir): os.mkdir(outdir)
    fh = open(logpath, 'r')
    res = []
    while True:
       aux = fh.readline()
       ckpt = int(aux.replace('model.ckpt-', '').replace('.meta', ''))
       print(ckpt)
       aux = fh.readline()
       idx = aux.find('class_0')
       aux = aux[idx+12:]
       idx = aux.find(']')
       aux = aux[:idx]
       iou0 = float(aux)

       aux = fh.readline()
       idx = aux.find('class_1')
       aux = aux[idx+12:]
       idx = aux.find(']')
       aux = aux[:idx]
       iou1 = float(aux)

       res.append([ckpt, iou0, iou1, (iou0+iou1)/2])

       if ckpt == 9730: break

    fh.close()

##########################################################
def read_hdf5(h5path):
    """Read @h5py
    """
    fh = h5py.File(h5path, "r")
    return np.array(fh['data'])
