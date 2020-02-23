#!/usr/bin/env python3
"""Count labels in a folder (each file contains a label -1,0,1,2)

python src/analyze_labels.py --annotdir ~/temp/20200124-sp20180511_imgs_sample2000_annot/ --imdir ~/results/graffiti/20200124-sp20180511_imgs_sample2000
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotdir', required=True, help='Annotation directory')
    # parser.add_argument('--imdir', required=True, help='Images directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    outdir = '/tmp/labels/'
    if os.path.exists(outdir):
        info('Path {} already exists'.format(outdir))
        return
    os.mkdir(outdir)

    files = sorted(os.listdir(args.annotdir))

    labels = '1,2,3'.split(',')
    nlabels = len(labels)
    info('Using labels:{}'.format(labels))
    elements = { l:[] for l in labels }

    nfiles = len(files)

    df_file = []
    df_xs = []
    df_ys = []
    df_labels = []

    i = 0
    for f in files:
        if not f.endswith('.txt'): continue
        filepath = pjoin(args.annotdir, f)
        arr = os.path.split(filepath)[-1].replace('.txt', '').split('_')
        # labelstr = open(filepath).read().strip().split(',')
        labels_ = open(filepath).read().strip().split(',')
        for l in labels_:
            df_file.append(f)
            df_ys.append(arr[1])
            df_xs.append(arr[2])
            df_labels.append(l)
            elements[l] += [os.path.split(filepath)[-1].replace('.txt', '.jpg')]
            i += 1

    df = pd.DataFrame({'filename': df_file, 'x': df_xs, 'y': df_ys, 'label': df_labels})
    df.to_csv(pjoin(outdir, 'labels.csv'), index_label='id',
              columns=['x', 'y', 'label'])

    fhs = {}
    for label in labels: # open
        listpath = os.path.join(outdir, 'label_{}.lst'.format(label))
        fhs[label] = open(listpath, 'w') 
        info('Creating file {}'.format(listpath))

    for label, elements in elements.items():
        info('label {}: {} elements'.format(label, len(elements)))
        fhs[label].write('\n'.join(elements))

    for label in labels: # close
        fhs[label].close()

    cmd = '# Replace <IMDIR> # export PREV=${{PWD}} && for I in {}; '\
        'do mkdir {}/label_${{I}} -p && '\
        'cd {} && '\
        'xargs --arg-file {}/label_${{I}}.lst '\
        'cp --target-directory="{}/label_${{I}}/"; done && '\
        'cd ${{PREV}}'. \
        format(' '.join(labels), outdir, '<IMDIR>', outdir, outdir)
    info('Check the target-directory prior to run it:')
    info(cmd)

if __name__ == "__main__":
    main()

