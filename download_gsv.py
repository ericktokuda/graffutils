#!/usr/bin/env python3
"""Download images from google street view
"""

import numpy as np
import requests
import os
import pandas as pd
from subprocess import Popen, PIPE
import argparse
import time
import datetime
import random

def get_metadata(coordsdf, apikey, outdir, samplesz):
    """Get metadata from coordinates in @coordsdf. Save them in outdir.

    Args:
    coordsdf(pandas.dataframe): dataframe with lat and lon fields
    outdir(str): output directory

    Returns:
    ret
    """
    print('Getting metadata...'); 
    urltemplate = 'https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location=XX,YY&fov=90&heading=0&pitch=20&key={}'.format(apikey)
    acc = 0
    
    for index, row in coordsdf.iterrows():
        if samplesz:
            if random.random() > samplesz: continue
        lat = float(row['lat'])
        lon = float(row['lon'])

        url = urltemplate.replace('XX', str(lat)).replace('YY', str(lon))
        filename = '_{}_{}.json'.format(round(float(lat), 8), round(float(lon), 8))
        outpath = os.path.join(outdir, filename)
        if os.path.exists(outpath): continue
        #input(url)
        print(url)
        code = download_and_store(url, outpath)

        if code != 200: return

        acc += 1
        if acc % 100 == 0: print('{} metadata downloaded'.format(acc))

def get_images(coordsdf, apikey, delay, outdir, samplesz=False):
    """Get images from coordinates in @coordsdf. Save them in outdir.

    Args:
    coordsdf(pandas.dataframe): dataframe with lat and lon fields
    outdir(str): output directory

    Returns:
    ret
    """

    logfile = '/tmp/' + '{}-images.log'.format(datetime.datetime.now().\
                                            strftime('%Y%m%d%H%m%s'))
    print('Getting images...'); 
    angles = [0 ,90, 180, 270]
    urltemplate = 'https://maps.googleapis.com/maps/api/streetview?size=640x640&location=XX,YY&fov=90&heading=ANG&pitch=20&key={}'.format(apikey)
    acc = 0
    consecerrors = 0
    
    for index, row in coordsdf.iterrows():
        if samplesz:
            if random.random() > samplesz: continue
        lat = round(float(row['lat']), 8)
        lon = round(float(row['lon']), 8)

        for ang in angles:
            aux = urltemplate.replace('XX', str(lat)).replace('YY', str(lon))
            url = aux.replace('ANG', str(ang))
            filename = '_{}_{}_{}.jpg'.format(lat, lon, ang)
            outpath = os.path.join(outdir, filename)
            if os.path.exists(outpath): continue
            print(url)
            code = download_and_store(url, outpath)

            if code == 200:
                consecerrors = 0
            elif code == 403:
                consecerrors += 1
            else:
                consecerrors = 0
                with open(logfile, 'a') as fh:
                    fh.write('{},{}\n'.format(url, code))
                time.sleep(0.1)

            if consecerrors > 3: return

            #time.sleep(random.random()*1)

            acc += 1
            if acc % 100 == 0: print('{} images downloaded'.format(acc))

def download_and_store(url, outpath):
    response = requests.get(url)
    if response.status_code == 200:
        with open(outpath, 'wb') as f:
            f.write(response.content)
    else:
        print('Return code:{}'.format(response))
    return response.status_code

def filtercoords(coords, toskip):
    for index, row in toskip.iterrows():
        x = (coords['lat'] == row['lat']) & (coords['lon'] == row['lon'])
        y = coords.index[x].tolist()
        #print(y)
        coords.drop(y, inplace=True)
    return coords

def args_ok(args):
    for arg in vars(args):
        if getattr(args, arg) == None: return False
    return True

##########################################################
def main():
    delay = 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', help='Output directory')
    parser.add_argument('--coordsfile', help='CSV containing the lat and lon' \
                        ' (with header) of the desired images.')
    parser.add_argument('--apikey', help='API key')
    parser.add_argument('--metadata', action='store_true', help='download metadata')
    parser.add_argument('--samplesz', required=False, default=False, help='sample size')
    args = parser.parse_args()

    if not args_ok(args):
        parser.print_help()
        return

    coords= pd.read_csv(args.coordsfile)

    if args.metadata:
        get_metadata(coords, args.apikey, args.outdir, args.samplesz)
    else:
        get_images(coords, args.apikey, delay, args.outdir)
        #get_images(coords, args.apikey, delay, args.outdir, 0.1)

##########################################################
if __name__ == "__main__":
    main()

