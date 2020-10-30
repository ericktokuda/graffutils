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
    if len(ax.shape) == 1: onerow = True
    elif len(ax.shape) == 2: onerow = False
    else: raise Exception('Just handle 1D or 2D axes')

    n = 1
    for el in ax.shape: n *= el

    for k in range(n):
        if onerow:
            ax[k].set_title('')
        else:
            i = k // ax.shape[1]
            j = k  % ax.shape[1]
            ax[i, j].set_title('')

    for k in range(n):
        coordsys = fig.dpi_scale_trans.inverted()

        if onerow:
            extent = ax[k].get_window_extent().transformed(coordsys)
        else:
            i = k // ax.shape[1]
            j = k  % ax.shape[1]
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


def generate_square_grid(minlat, maxlat, minlon, maxlon, delta, outfile):
    """Generate grid and output to file

    Args:
    minlat(float): min latitude
    maxlon(float): max latitude
    minlon(float): min longitude
    maxlon(float): max longitude
    """
    fh = open(outfile, 'w')
    d = delta
    nlat = int((maxlat - minlat)/d) + 1
    nlon = int((maxlon - minlon)/d) + 1

    fh.write('id,lat,lon\n')
    id = 1
    for i in range(nlat):
        lat = round(minlat + i*d, 8)
        for j in range(nlon):
            lon = round(minlon + j*d, 8)
            fh.write('{},{:.8f},{:.8f}\n'.format(id, lat, lon))
            id += 1

    fh.close()
