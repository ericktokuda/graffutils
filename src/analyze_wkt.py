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

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def get_areas_from_wkt(wktpath):
    wktfh = open(wktpath)
    shapelyobj = shapely.wkt.loads(wktfh.read())
    areas = []
    for poly in shapelyobj:
        areas.append(poly.area)
    wktfh.close()
    return sorted(areas)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('--wktdir', required=True, help='Wkt directory')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    pref = '_40.48892421_-3.67150833_'
    get_areas_from_location(latstr, lon)
    wktpath = pjoin(os.getenv('HOME'), 'temp/20200408-graffiti_sample/madrid_wkt/wkt/_40.48892421_-3.67150833_270.wkt')
    areas = get_areas_from_wkt(wktpath)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

