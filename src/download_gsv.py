#!/usr/bin/env python3
"""Download gsv image if metadata is available
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
from myutils import info, create_readme
import pandas as pd
import requests
import json

KEY = os.environ['GOOGLEKEY']
TEMPLATE = 'https://maps.googleapis.com/maps/api/streetviewOPTION?size=640x640&location=LAT,LON&fov=90&heading=ANG&pitch=20&key=' + KEY

##########################################################
def download_images(lon, lat, outdir, ntries=3):
    """Download metadata and return contents"""
    info(inspect.stack()[0][3] + '()')
    angles = [0 ,90, 180, 270]
    url = TEMPLATE.replace('OPTION', '')
    urlorig = url.replace('LON', str(lon)).replace('LAT', str(lat))

    nimgs = 0
    for ang in angles:
        url = urlorig.replace('ANG', str(ang))
        outpath = pjoin(outdir, '_{:.08f}_{:.08f}_{}.jpg'.format(lat, lon, ang))
        if os.path.exists(outpath): continue
        for it in range(ntries):
            info('url:{}'.format(url))
            
            r = requests.get(url)
            if r.status_code != 200: # conectivity issue
                code = str(r.status_code)
                time.sleep(np.random.rand()*5)
                continue

            open(outpath, 'wb').write(r.content)
            nimgs += 1
            break
    return nimgs

##########################################################
def get_metadata(lon, lat, outdir, ntries=3):
    """Download metadata and return contents"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, '_{:.08f}_{:.08f}.json'.format(lat, lon))

    try:
        z = json.load(open(outpath))
        if 'location' in z:
            return z['status'], z['location']['lng'], z['location']['lat']
        else:
            return z['status'], lon, lat
    except Exception as e:
        pass

    url = TEMPLATE.replace('OPTION', '/metadata')
    url = url.replace('LON', str(lon)).replace('LAT', str(lat)).replace('ANG', '0')
    info('url:{}'.format(url))

    metadata = []

    lonsnap = lon; latsnap = lat;
    for it in range(ntries):
        r = requests.get(url)
        if r.status_code != 200: # conectivity issue
            code = str(r.status_code)
            time.sleep(np.random.rand()*5)
            continue

        with open(outpath, 'wb') as f:
            f.write(r.content)

        code = r.json()['status']
        if code == 'OK':
            lonsnap = r.json()['location']['lng']
            latsnap = r.json()['location']['lat']
            if not ('Google' in r.json()['copyright']): code = 'NONGOOGLE'
        break

    return code, lonsnap, latsnap

##########################################################
def download_all(df, quota, outdir):
    """Download metadata and image, if available """
    info(inspect.stack()[0][3] + '()')

    metadatadir = pjoin(outdir, 'metadata')
    if not os.path.isdir(metadatadir): os.mkdir(metadatadir)
    imgdir = pjoin(outdir, 'img')
    if not os.path.isdir(imgdir): os.mkdir(imgdir)

    downloads = []

    acc = 0
    # for idx, row in df.iterrows():
    # for idx, row in df[::-1].iterrows():
    for idx, row in df[::-1].iterrows():
        if idx % 3 != 0: continue
        info('idx:{}'.format(idx))

        statusmeta, lonsnap, latsnap = get_metadata(row.lon,
                row.lat, metadatadir)

        if statusmeta == 'OK':
            numimg = download_images(lonsnap, latsnap, imgdir)
            acc += numimg

        downloads.append([row.lon, row.lat, lonsnap, latsnap, statusmeta, numimg])
        if acc > int(quota): break

    cols = 'lon,lat,lonsnap,latsnap,statusmeta,numimg'.split(',')
    df = pd.DataFrame(downloads, columns=cols)
    df.to_csv(pjoin(outdir, 'downloads.csv'))

##########################################################
def print_warning(quota):
    print('Quota is set to {}. Please check the current pricing:'.format(quota))
    print('https://developers.google.com/maps/documentation/streetview/usage-and-billing\n')
    print('and the available quota in the free tier:')
    print('https://console.cloud.google.com/billing/\n')
    print('Also, the key provided is', KEY)
    print('If you are sure about these info\n')
    print('press any key to continue')

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coords', required=True, help='Csv with lat and lon')
    parser.add_argument('--quota', default=10, help='Quota available (Check our monthly quota available in Google cloud prior to execute)')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    readmepath = create_readme(sys.argv, args.outdir)

    print_warning(args.quota); input()

    df = pd.read_csv(args.coords)
    download_all(df, args.quota, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
