#!/usr/bin/env python3
"""Utility functions for the graffiti project

# featureextractor
rm /tmp/out -rf && python src/utils.py featureextractor ~/results/graffiti/20200101-deeplab/20200109-sp20180511_crops/crop/20180511-gsv_spcity --outdir /tmp/out

# featuresummarizer
rm /tmp/out -rf && python src/utils.py featuresummarizer ~/results/graffiti/20200101-deeplab/20200109-sp20180511_crops/feature/20180511-gsv_spcity/ all --outdir /tmp/out

# masksgenerator
rm /tmp/out -rf && python src/utils.py masksgenerator ~/results/graffiti/20200101-deeplab/20200109-sp20180511_crops/wkt/20180511-gsv_spcity/  /media/frodo/6TB_A/gsvcities/20180511-gsv_spcity/img --outdir /tmp/out

# infomapparser
rm /tmp/out -rf && python src/utils.py infomapparser ~/results/graffiti/20200222-citysp_infomap.clu ~/results/graffiti/20200221-citysp.graphml  ~/results/graffiti/20200209-cityspold_8003_annot_labels/labels.csv --outdir /tmp/out

# labelshuffler
rm /tmp/shuffle -rf && python src/utils.py labelshuffler ~/results/graffiti/20200209-cityspold_8003_labels_clu.csv  --outdir /tmp/shuffle

# labelplotter  - generate plots for the article
rm /tmp/foo -rf && python src/utils.py labelplotter ~/results/graffiti/20200209-cityspold_8003_labels_clu.csv ~/results/graffiti/20200222-citysp_infomap_areas.csv --outdir /tmp/foo

# mapgenerator - generate a map of the counts
rm /tmp/foo -rf && python src/utils.py mapgenerator ~/results/graffiti/20200202-types/20200221-citysp.graphml ~/results/graffiti/20200202-types/20200224-citysp_shp/ ~/results/graffiti/20200202-types/20200209-cityspold_8003_labels_clu.csv ~/results/graffiti/20200202-types/20200222-citysp_infomap.clu --outdir /tmp/foo
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
import pandas as pd

from src import clustering
from src import labelling
from src import featextraction

HOME = os.getenv('HOME')

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    annotdir = pjoin(HOME, 'results/graffiti/20200202-types/20200209-sample_8003_annot/')
    outdir = args.outdir
    # featpath = pjoin(HOME, 'results/graffiti/' \
            # '20200202-types/20200115-features_resnet18_sample1000.csv')
    # labelling.summarize_from_dir(annotdir, outdir)
    # clustering.cluster(featpath, 'all', outdir)
    # featextraction.extract_features_all(imdir, outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
