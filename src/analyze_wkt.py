#!/usr/bin/env python3
"""Analyze masks in wkt format obtained using deeplab
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
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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
def pandas2geopandas(df):
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    crs = {'init': 'epsg:4326'}
    geodf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    return geodf

##########################################################
def get_geodataframe(df, geodfpath):
    info(inspect.stack()[0][3] + '()')
    geodf = pandas2geopandas(df)
    geodf.to_file(driver='ESRI Shapefile', filename=geodfpath)
    return geodf

##########################################################
def plot_affected_areas(geodf, outdir):
    filtered = geodf[geodf.arealarge > 0]
    filtered[filtered.nlarge > 5].plot(column='arealarge', s=5, legend=True)
    plt.savefig(pjoin(outdir, 'areas.pdf'))

##########################################################
def get_wktdir_summary(wktdir, minarea, outdir):
    info(inspect.stack()[0][3] + '()')
    geodfpath = pjoin(outdir, 'areas_min{}.shp'.format(minarea))
    if os.path.exists(geodfpath): return gpd.read_file(geodfpath)

    df = parse_wktdir(wktdir, minarea, outdir)
    return get_geodataframe(df, geodfpath)

##########################################################
def parse_wktdir(wktdir, minarea, outdir):
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'areas_min{}.csv'.format(minarea))

    if os.path.exists(outpath):
        lastline = subprocess.check_output(['tail', '-1', outpath]).decode('UTF-8')
        arr = lastline.strip().split(',')
        if len(arr) > 0: startingidx = int(arr[0]) + int(arr[3])
    else:
        open(outpath, 'w').write('idx,lat,lon,nviews,nsmall,nlarge,arealarge\n')
        startingidx = 0

    files = sorted(os.listdir(wktdir))
    outfh = open(outpath, 'a')

    i = startingidx
    while i < len(files):
        info('i:{}'.format(i))
        f = files[i]
        if not f.endswith('.wkt'): i += 1; continue
        _, latstr, lonstr, _ = f.split('_')
        allareas, nviews, nsmall = parse_areas_from_wkts(latstr, lonstr,
                wktdir, minarea)
        outfh.write('{},{},{},{},{},{},{}\n'.format(i, latstr, lonstr, nviews,
            nsmall, len(allareas), int(np.sum(allareas))))
        i += nviews

    outfh.close()
    return pd.read_csv(outpath)

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
    minarea = 900

    geodf = get_wktdir_summary(args.wktdir, minarea, args.outdir)
    plot_affected_areas(geodf, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

