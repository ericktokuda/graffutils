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
    # info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, '_{:.08f}_{:.08f}.json'.format(lat, lon))

    try:
        existing = True
        z = json.load(open(outpath))
        if 'location' in z:
            return z['status'], z['location']['lng'], z['location']['lat'], existing
        else:
            return z['status'], lon, lat, existing
    except Exception as e:
        pass


    existing = False

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

    return code, lonsnap, latsnap, existing

##########################################################
def download_missing_metadata(imdir, cachedir):
    """Download metadata and image, if available """
    info(inspect.stack()[0][3] + '()')

    downloads = []

    files = sorted(os.listdir(imdir))
    nongoogle = 0
    for i, f in enumerate(files):
        info('i:{}'.format(i))
        _, latstr, lonstr, _ = f.strip().split('.jpg')[0].split('_')
        statusmeta, _, _, _ = get_metadata(float(lonstr), float(latstr), cachedir)
        if statusmeta != 'OK': nongoogle += 1
    info('nongoogle:{}'.format(nongoogle))

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
    ids = sorted(range(len(df)))

    for idx in ids:
        info('idx:{}'.format(idx))
        row = df.loc[idx]

        statusmeta, lonsnap, latsnap, _ = get_metadata(row.lon,
                row.lat, metadatadir)

        if statusmeta == 'OK':
            numimg = download_images(lonsnap, latsnap, imgdir)
            acc += numimg
        else: numimg = 0

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

    # download_missing_metadata('madrid/img/', 'madrid/newmetadata/')
    print_warning(args.quota); input()

    df = pd.read_csv(args.coords)
    download_all(df, args.quota, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
