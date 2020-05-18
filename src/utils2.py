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
def read_hdf5(h5path):
    """Read @h5py
    """
    fh = h5py.File(h5path, "r")
    return np.array(fh['data'])
