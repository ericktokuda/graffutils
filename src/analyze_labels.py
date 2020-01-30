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
    parser.add_argument('--imdir', required=True, help='Images directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    outdir = '/tmp/labels/'
    if os.path.exists(outdir):
        info('Path {} already exists'.format(outdir))
        return
    os.mkdir(outdir)

    files = sorted(os.listdir(args.annotdir))

    labels = '-1,1,2,3'.split(',')
    info('Using labels:{}'.format(labels))
    elements = { l:[] for l in labels }

    for f in files:
        if not f.endswith('.txt'): continue
        filepath = pjoin(args.annotdir, f)
        labelstr = open(filepath).read().strip()
        elements[labelstr] += [os.path.split(filepath)[-1].replace('.txt', '.jpg')]

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

    cmd = 'export PREV=${{PWD}} && for I in {}; '\
        'do mkdir {}/label_${{I}} -p && '\
        'cd {} && '\
        'xargs --arg-file {}/label_${{I}}.lst '\
        'cp --target-directory="{}/label_${{I}}/"; done && '\
        'cd ${{PREV}}'. \
        format(' '.join(labels), outdir, args.imdir, outdir, outdir)
    info('Check the target-directory prior to run it:')
    info(cmd)

if __name__ == "__main__":
    main()

