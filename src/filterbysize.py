#!/usr/bin/env python3
"""Filter images by size
"""

import argparse
import logging
import time
from os.path import join as pjoin
from logging import debug, info
import pandas as pd
import os

class FilterBySize:
    def __init__(self):
        pass

    def run(self, args, outdir):
        if len(args) < 1:
            info('Please provide (1)the resolutions file as argument')
            info('Generate it with:\n{}'.format('''find . -iname "*.jpg" -type f -exec identify -format '%i,%w,%h\n' '{}' \; > /tmp/res.csv'''))
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args[0]))
            return

        respath = args[0]

        df = pd.read_csv(respath, header=None, names=['file', 'w', 'h'])
        orignlines = len(df)
        small = df[(df.w < 20) | (df.h < 20)]
        info('Before filtering: {}, after filtering:{}'.format(orignlines, len(small)))
        small.file.to_csv(pjoint(outdir, 'todel.csv'), index=False)
