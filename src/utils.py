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

##########################################################
def export_individual_axis(ax, fig, labels, outdir, pad=0.3, prefix='', fmt='pdf'):
    n = ax.shape[0]*ax.shape[1]
    for k in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        ax[i, j].set_title('')

    for k in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        coordsys = fig.dpi_scale_trans.inverted()
        extent = ax[i, j].get_window_extent().transformed(coordsys)
        x0, y0, x1, y1 = extent.extents

        if isinstance(pad, list):
            x0 -= pad[0]; y0 -= pad[1]; x1 += pad[2]; y1 += pad[3];
        else:
            x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;

        bbox =  matplotlib.transforms.Bbox.from_extents(x0, y0, x1, y1)
        fig.savefig(pjoin(outdir, prefix + labels[k] + '.' + fmt),
                      bbox_inches=bbox)

##########################################################
def hex2rgb(hexcolours, normalized=False, alpha=None):
    rgbcolours = np.zeros((len(hexcolours), 3), dtype=int)
    for i, h in enumerate(hexcolours):
        rgbcolours[i, :] = np.array([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])

    if alpha != None:
        aux = np.zeros((len(hexcolours), 4), dtype=float)
        aux[:, :3] = rgbcolours / 255
        aux[:, -1] = alpha # alpha
        rgbcolours = aux
    elif normalized:
        rgbcolours = rgbcolours.astype(float) / 255

    return rgbcolours
