#!/usr/bin/env python3
"""Analyze WKTs generated from the masks from deeplab
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
import shapely.wkt
import subprocess

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def get_areas_from_wkt(wktpath):
    content = open(wktpath).read()
    if content == '': return []
    shapelyobj = shapely.wkt.loads(content)
    areas = []
    for poly in shapelyobj:
        areas.append(poly.area)
    return sorted(areas)

##########################################################
def parse_areas_from_wkts(latstr, lonstr, wktdir, minarea):
    pref = '_{}_{}_'.format(latstr, lonstr)
    headings = '0,90,180,270'.split(',')
    pref = pjoin(wktdir, '_{}_{}_'.format(latstr, lonstr))

    allareas = []
    nviews = 0; nsmall = 0
    for heading in headings:
        wktpath = pref + heading + '.wkt'
        try:
            areas = np.array(get_areas_from_wkt(wktpath))
        except Exception as e:
            continue
        nsmall += np.sum(areas < minarea)
        areas = areas[areas >= minarea]
        nviews += 1
        allareas.extend(areas)
    return allareas, nviews, nsmall

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wktdir', required=True, help='Wkt directory')
    parser.add_argument('--minarea', type=int, default=900, help='Wkt directory')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    files = sorted(os.listdir(args.wktdir))

    outpath = pjoin(args.outdir, 'areas_min{}.csv'.format(args.minarea))

    startingidx = 0
    if os.path.exists(outpath):
        lastline = subprocess.check_output(['tail', '-1', outpath]).decode('UTF-8')
        arr = lastline.strip().split(',')
        if len(arr) > 0: startingidx = int(arr[0]) + int(arr[3])
    else:
        open(outpath, 'w').write('idx,lat,lon,nviews,nsmall,nlarge,arealarge\n')

    outfh = open(outpath, 'a')

    i = startingidx
    while i < len(files):
        info('i:{}'.format(i))
        f = files[i]
        if not f.endswith('.wkt'): i += 1; continue
        _, latstr, lonstr, _ = f.split('_')
        allareas, nviews, nsmall = parse_areas_from_wkts(latstr, lonstr, args.wktdir,
                args.minarea)
        outfh.write('{},{},{},{},{},{},{}\n'.format(i, latstr, lonstr, nviews,
            nsmall, len(allareas), int(np.sum(allareas))))
        i += nviews

    outfh.close()

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

