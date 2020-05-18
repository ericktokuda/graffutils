#!/usr/bin/env python3
"""Utility functions for the graffiti project

# labelanalyzer
rm /tmp/out -rf && python src/utils.py labelanalyzer /tmp /tmp/ --outdir /tmp/out

# clustering
rm /tmp/out -rf && python src/utils.py clustering all ~/results/graffiti/20200115-features_resnet18_sample1000.csv /tmp/ --outdir /tmp/out

# filterbysize
rm /tmp/out -rf && python src/utils.py filterbysize ~/results/graffiti/20200101-deeplab/20200109-sp20180511_crops/feature/20180511-gsv_spcity/ --outdir /tmp/out                    

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
import logging
import time
from os.path import join as pjoin
from logging import debug, info
import os
import numpy as np
import pandas as pd
import pickle as pkl


# from labelanalyzer import LabelAnalyzer
# from clustering import Clustering
from filterbysize import FilterBySize
from featureextractor import FeatureExtractor
from featuresummarizer import FeatureSummarizer
from masksgenerator import MasksGenerator
from deeplabanalyzer import DeeplabAnalyzer
from labelshuffler import LabelShuffler
from infomapparser import InfomapParser
from labelplotter import LabelPlotter
from mapgenerator import MapGenerator

##########################################################
def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('args', nargs='*', help='1: function 2-*: function arguments')
    parser.add_argument('--outdir', default='/tmp/', help='Output dir')
    arguments = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if os.path.exists(arguments.outdir):
        info('Path {} already exists...'.format(arguments.outdir))
    else:
        os.mkdir(arguments.outdir)
        # return


    options = dict(
        # labelanalyzer = LabelAnalyzer,
        # clustering = Clustering,
        filterbysize = FilterBySize,
        featureextractor = FeatureExtractor,
        featuresummarizer = FeatureSummarizer,
        deeplabanalyzer = DeeplabAnalyzer,
        labelshuffler = LabelShuffler,
        masksgenerator = MasksGenerator,
        infomapparser = InfomapParser,
        labelplotter = LabelPlotter,
        mapgenerator = MapGenerator,
    )

    if len(arguments.args) == 0:
        info('Please provide the function you want to call as the first argument\n'\
             'or call this script with --help. Aborting...')
        return

    if arguments.args[0] in options.keys():
        exps = options[arguments.args[0]]()
        exps.run(arguments.args[1:], arguments.outdir)
    else:
        info('Please choose one among {}. Aborting...'.format(options.keys()))

    info('Elapsed time:{}'.format(time.time()-t0))

if __name__ == "__main__":
    main()
