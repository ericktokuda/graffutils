#!/usr/bin/env python3
"""Validate Liberdade images
ALREADY INCLUDED IN UTILS.PY
"""

import os
import pandas as pd

def validate_metadata(metadatadir, coordsfile):
    fh = open(coordsfile)
    files = {}
    for l in fh:
        lon, lat = [ float(x) for x in l.strip().split(',')]
        files.add('im_{:8f}_{:8f}_0.json'.format(lat, lon))
    fh.close()
    print(files)

def validate_files(imgdir, coordsfile):
    df = pd.read_csv(coordsfile)
    lon = set(df['lon'])
    lat = set(df['lat'])
    for f in os.listdir(imgdir):
        if not f.endswith('.jpg'):
            print('{} not jpeg'.format(f))
            return
        arr = f.split('_')
        flat = float(arr[1])
        flon = float(arr[2])

        if not flat in lat or not flon in lon:
            print(f)
            continue

        #print(len(df[(df['lat'] == flat) & (df['lon'] == flon)]))


##########################################################
def main():
    imgdir = '/home/keiji/results/graffiti/20180511-gsv_liberdade/img/'
    metadatadir = '/home/keiji/results/graffiti/20180511-gsv_liberdade/metadata/'
    coordsfile = '/home/keiji/results/graffiti/20180511-gsv_liberdade/points0001liberdade.csv'

    validate_metadata(metadatadir, coordsfile)
    #validate_files(imgdir, coordsfile)


if __name__ == "__main__":
    main()

