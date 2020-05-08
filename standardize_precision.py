#!/usr/bin/env python3
"""Standardize precision

"""

import argparse
import shutil
import os

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Input csv')
    args = parser.parse_args()

    #standardize_csv(args.input)
    standardize_dir(args.input)

def standardize_dir(indir):
    for f in os.listdir(indir):
        if not 'im' in f: continue
        arr = f.split('_')
        #print(arr)
        latstr = '{:8f}'.format(float(arr[1]))
        lonstr = '{:8f}'.format(float(arr[2]))
        newf = 'im_{:8f}_{:8f}_0.json'.format(float(arr[1]), float(arr[2]))
        shutil.copy(os.path.join(indir, f), '/tmp/' + newf)

def standardize_csv(incsv):
    fh = open(args.incsv)
    print(fh.readline().strip())
    for l in fh:
        lon, lat = l.strip().split(',')
        print('{:8f},{:8f}'.format(float(lon), float(lat)))

    fh.close()


if __name__ == "__main__":
    main()

