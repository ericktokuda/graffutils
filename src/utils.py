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
import datetime
import pandas as pd
import sys
import h5py
import inspect
from myutils import info, create_readme
import json

#############################################################
def rename_8digplaces(outdir):
    dirs = [] # just for safety

    i = 0
    for d in dirs:
        info('d:{}'.format(d))
        with open('/tmp/processed.txt', 'a') as fh: fh.write(d + '\n')
        for f in sorted(os.listdir(d)):
            if not f.endswith('.jpg'): continue
            _, lat, lon, ang = f.split('.jpg')[0].split('_')
            fixed = '_{:.08f}_{:.08f}_{}.jpg'.format(float(lat), float(lon), ang)
            if f != fixed:
                info('i:{}'.format(i))
                os.rename(pjoin(d, f), pjoin(d, fixed))
                i += 1
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

##########################################################
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

##########################################################
def compile_metadata(metadatadir, outdir):
    """Compile all the metadata into a single csv"""
    info(inspect.stack()[0][3] + '()')
    files = os.listdir(metadatadir)
    curdir = os.getcwd(); os.chdir(metadatadir)
    data = []
    zerores = []
    unknnerr = []
    notgoogle = []
    for i, f in enumerate(files):
        if i % 100 == 0: info('{} processed files'.format(i))
        if not f.endswith('.json') or (not os.path.getsize(f)):
            info('Issue with:{}'.format(f))
            continue
        js = json.load(open(f, 'r'))

        if js['status'] == 'ZERO_RESULTS':
            zerores.append(f)
            continue
        elif js['status'] == 'UNKNOWN_ERROR':
            unknnerr.append(f)
            continue
        elif not ('copyright' in js.keys()) or (not 'Google' in js['copyright']):
            notgoogle.append(f)
            continue

        capturedon = js['date'] if 'date' in js.keys() else ''
        _, latgrid, longrid = f.replace('.jpg', '').split('_')
        lat = js['location']['lat']
        lon = js['location']['lng']
        panoid = js['pano_id']
        status = js['status']
        assert js['status'] == 'OK'
        row = [lat, lon, latgrid, longrid, capturedon, panoid]
        data.append(row)

    os.chdir(curdir)
    colnames = ['lat', 'lon', 'latgrid', 'longrid', 'capturedon', 'panoid']
    open(pjoin(outdir, 'metadata_zerores.lst'), 'w').write('\n'.join(zerores))
    open(pjoin(outdir, 'metadata_unknerr.lst'), 'w').write('\n'.join(unknnerr))
    open(pjoin(outdir, 'metadata_notgoogle.lst'), 'w').write('\n'.join(notgoogle))
    df = pd.DataFrame(data, columns=colnames)
    df.to_csv(pjoin(outdir, 'metadata_ok.csv'), index=False)

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    metadatadir = '/home/frodo/datasets/gsvcities/spac0005/nyc/metadata/'
    compile_metadata(metadatadir, outdir)


##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
