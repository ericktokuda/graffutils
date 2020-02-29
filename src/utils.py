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
rm /tmp/out -rf && python src/utils.py infomapparser ~/results/graffiti/20200221-citysp.clu ~/temp/citysp.graphml  ~/results/graffiti/20200209-sample_8003_annot_labels/labels.csv --outdir /tmp/out

# labelshuffler
rm /tmp/shuffle -rf && python src/utils.py labelshuffler /tmp/out/clusters.csv   --outdir /tmp/shuffle
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


from labelanalyzer import LabelAnalyzer
from clustering import Clustering
from filterbysize import FilterBySize
from featureextractor import FeatureExtractor
from featuresummarizer import FeatureSummarizer
from masksgenerator import MasksGenerator
from deeplabanalyzer import DeeplabAnalyzer
from labelshuffler import LabelShuffler
from infomapparser import InfomapParser

##########################################################
def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('args', nargs='*', help='1: function 2-*: function arguments')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output dir')
    arguments = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if os.path.exists(arguments.outdir):
        info('Path {} already exists. Aborting...'.format(arguments.outdir))
        return

    os.mkdir(arguments.outdir)

    options = dict(
        labelanalyzer = LabelAnalyzer,
        clustering = Clustering,
        filterbysize = FilterBySize,
        featureextractor = FeatureExtractor,
        featuresummarizer = FeatureSummarizer,
        deeplabanalyzer = DeeplabAnalyzer,
        labelshuffler = LabelShuffler,
        infomapparser = InfomapParser,
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

###########################################################
###########################################################
###########################################################
#"""Plot input graphml
#"""
#
#import igraph
#import matplotlib.pyplot as plt
#import networkx as nx
#import colorsys
#
#def generate_rgb_colors(n):
#    hsvcols = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
#    rgbcols = map(lambda x: colorsys.hsv_to_rgb(*x), hsvcols)
#    return list(rgbcols)
#
#def main():
#    parser = argparse.ArgumentParser(description=__doc__)
#    parser.add_argument('--graphml', required=True, help='Graphml path')
#    args = parser.parse_args()
#
#    logging.basicConfig(format='[%(asctime)s] %(message)s',
#    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)
#    g = igraph.Graph.Read_GraphML(args.graphml)
#    g.to_undirected()
#    info('Loaded')
#    if 'x' in g.vs.attributes():
#        g.vs['x'] = np.array(g.vs['x']).astype(float)
#        g.vs['y'] = -np.array(g.vs['y']).astype(float)
#
#    info('g.vcount():{}'.format(g.vcount()))
#    info('g.ecount():{}'.format(g.ecount()))
#    del g.vs['id'] # Avoid future warnings
#    attrs = g.es.attributes()
#    for attr in attrs: del g.es[attr]
#    igraph.write(g, '/tmp/foo.graphml', 'graphml')
#    info('saved')
#    info('Reload')
#    g = igraph.read('/tmp/foo.graphml')
#    g.simplify()
#    # dendr = g.community_fastgreedy()
#    # clusters = dendr.as_clustering()
#
#    # clusters = g.community_leading_eigenvector(clusters=5)
#    # clusters = g.community_edge_betweenness(clusters=5) # get stucked
#    dendr = g.community_walktrap()
#    clusters = dendr.as_clustering()
#    # get the membership vector
#    membership = clusters.membership
#    colorlist = generate_rgb_colors(len(np.unique(membership)))
#
#    igraph.plot(g, target='/tmp/communities.pdf', vertex_size=4,
#                bbox = (2000, 2000),
#                vertex_color=[ colorlist[m] for m in membership ],
#                vertex_frame_width=0)
#
#
#
###########################################################
#""" Evaluate with frozen model
#
#export CUDA_VISIBLE_DEVICES=1
#nohup python deeplab_demo.py --frozenpath ~/temp/frozen_inference_graph-20868-multiscale.pb --dirslist ~/temp/gsvcities_dirs.lst --outdir ~/temp/20191221-gsvcities_polygons/ 2>&1 > ~/temp/20191221-gsvcities_wkt.log &
#"""
#
#import sys
#import cv2
#import shapely
#import shapely.wkt
#from shapely.geometry import Polygon, MultiPolygon
#from pathlib import Path
#import random
#
#
#
#from io import BytesIO
#import tarfile
#import tempfile
#from six.moves import urllib
#
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
#from PIL import Image
#
#import tensorflow as tf
#
#class DeepLabModel(object):
#  """Class to load deeplab model and run inference."""
#
#  INPUT_TENSOR_NAME = 'ImageTensor:0'
#  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
#  INPUT_SIZE = 640
#
#  def __init__(self, frozenpath):
#    """Creates and loads pretrained deeplab model."""
#    self.graph = tf.Graph()
#
#    graph_def = None
#
#    file_handle  = open(frozenpath, 'rb')
#    graph_def = tf.GraphDef.FromString(file_handle.read())
#
#    if graph_def is None:
#      raise RuntimeError('Cannot find inference graph in tar archive.')
#
#    with self.graph.as_default():
#      tf.import_graph_def(graph_def, name='')
#
#    self.sess = tf.Session(graph=self.graph)
#
#  def run(self, image):
#    """Runs inference on a single image.
#
#    Args:
#      image: A PIL.Image object, raw input image.
#
#    Returns:
#      seg_map: Segmentation map of `resized_image`.
#    """
#    batch_seg_map = self.sess.run(
#        self.OUTPUT_TENSOR_NAME,
#        feed_dict={self.INPUT_TENSOR_NAME: [image]})
#    seg_map = batch_seg_map[0]
#    return seg_map
#
#
#def create_pascal_label_colormap():
#  """Creates a label colormap used in PASCAL VOC segmentation benchmark.
#
#  Returns:
#    A Colormap for visualizing segmentation results.
#  """
#  colormap = np.zeros((256, 3), dtype=int)
#  ind = np.arange(256, dtype=int)
#
#  for shift in reversed(range(8)):
#    for channel in range(3):
#      colormap[:, channel] |= ((ind >> channel) & 1) << shift
#    ind >>= 3
#
#  return colormap
#
#
#def label_to_color_image(label):
#  """Adds color defined by the dataset colormap to the label.
#
#  Args:
#    label: A 2D array with integer type, storing the segmentation label.
#
#  Returns:
#    result: A 2D array with floating type. The element of the array
#      is the color indexed by the corresponding element in the input label
#      to the PASCAL color map.
#
#  Raises:
#    ValueError: If label is not of rank 2 or its value is larger than color
#      map maximum entry.
#  """
#  if label.ndim != 2:
#    raise ValueError('Expect 2-D input label')
#
#  colormap = create_pascal_label_colormap()
#
#  if np.max(label) >= len(colormap):
#    raise ValueError('label value too large.')
#
#  return colormap[label]
#
#
#def vis_segmentation(image, seg_map):
#  """Visualizes input image, segmentation map and overlay view."""
#  plt.figure(figsize=(15, 5))
#  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
#
#  plt.subplot(grid_spec[0])
#  plt.imshow(image)
#  plt.axis('off')
#  plt.title('input image')
#
#  plt.subplot(grid_spec[1])
#  seg_image = label_to_color_image(seg_map).astype(np.uint8)
#  plt.imshow(seg_image)
#  plt.axis('off')
#  plt.title('segmentation map')
#
#  plt.subplot(grid_spec[2])
#  plt.imshow(image)
#  plt.imshow(seg_image, alpha=0.7)
#  plt.axis('off')
#  plt.title('segmentation overlay')
#
#  unique_labels = np.unique(seg_map)
#  ax = plt.subplot(grid_spec[3])
#  plt.imshow(
#      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
#  ax.yaxis.tick_right()
#  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
#  plt.xticks([], [])
#  ax.tick_params(width=0.0)
#  plt.grid('off')
#  plt.show()
#
#
#
## def get_2dpoints_from_cv2_struct(cv2points):
#    # points = []
#    # for cv2point in cv2points:
#        # points.append(cv2point[0])
#    # return points
#
#def get_contours(mask_):
#    epsilon = 2
#    aux, _ = cv2.findContours(mask_.astype(np.uint8), cv2.RETR_EXTERNAL,
#                              cv2.CHAIN_APPROX_SIMPLE)
#    contours = []
#    for cnt in aux:
#        aux2 = cv2.approxPolyDP(cnt, epsilon, True)
#        if len(aux2) < 3: continue
#        # aux3 = get_2dpoints_from_cv2_struct(aux2)
#        aux3 = aux2
#        contours.append(aux3)
#    return contours
#
###########################################################
#def dump_contours_to_wkt(polys, wktpath):
#    if len(polys) == 0:
#        Path(wktpath).touch()
#        return
#
#    # print('wktpath:{}'.format(wktpath))
#    shapelyinput = []
#
#    for poly in polys:
#        points = [ list(p[0]) for p in poly ]
#        shapelyinput.append([points,[]])
#
#    shapelycontours = MultiPolygon(shapelyinput)
#    with open(wktpath, 'w') as fh:
#        fh.write(shapelycontours.wkt)
#
#def crop_masks(im, imfilename, polys, cropdir):
#    if len(polys) == 0: return
#
#    shapelyinput = []
#
#    for i, poly in enumerate(polys):
#        croppath = pjoin(cropdir,
#                        '{}_{:02d}.jpg'. \
#                        format(os.path.splitext(imfilename)[0], i))
#        coords = np.squeeze(poly)
#        xmin, ymin = np.min(coords, 0)
#        xmax, ymax = np.max(coords, 0)
#
#        if xmax - xmin < 10 or ymax - ymin < 10 : continue # Too small
#
#        cropped = im.crop((xmin, ymin, xmax, ymax))
#        cropped.save(croppath)
###########################################################
#def run_visualization(impath):
#  """Inferences DeepLab model and visualizes result."""
#
#  original_im = Image.open(impath)
#  resized_im, seg_map = MODEL.run(original_im)
#
#  vis_segmentation(resized_im, seg_map)
#
#def main():
#    parser = argparse.ArgumentParser(description=__doc__)
#    parser.add_argument('--frozenpath', required=True, help='Frozen model path')
#    parser.add_argument('--dirslist', required=True, help='List of the dirs containing the images')
#    parser.add_argument('--shuffle', action='store_true', help='Output directory')
#    parser.add_argument('--outdir', required=True, help='Output directory')
#    args = parser.parse_args()
#
#    logging.basicConfig(format='[%(asctime)s] %(message)s',
#    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)
#
#    LABEL_NAMES = np.asarray([
#        'background', 'tag', 'frame', 'sign',
#    ])
#
#    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
#    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
#
#    MODEL = DeepLabModel(args.frozenpath)
#    info('Model loaded successfully!')
#
#    os.makedirs(args.outdir, exist_ok=True)
#    os.makedirs(pjoin(args.outdir, 'wkt'), exist_ok=True)
#    os.makedirs(pjoin(args.outdir, 'masksum'), exist_ok=True)
#
#    dirpaths = open(args.dirslist).read().splitlines()
#    print('dirpaths:{}'.format(dirpaths))
#
#    for dir_ in dirpaths:
#        imdir = pjoin(dir_, 'img/')
#        info('{}'.format(dir_))
#
#        lastpath = os.path.basename(os.path.normpath(dir_))
#        wktdir = pjoin(args.outdir, 'wkt', lastpath)
#        os.makedirs(wktdir, exist_ok=True)
#
#        masksumdir = pjoin(args.outdir, 'masksum', lastpath)
#        os.makedirs(masksumdir, exist_ok=True)
#
#        cropdir = pjoin(args.outdir, 'crop', lastpath)
#        os.makedirs(cropdir, exist_ok=True)
#
#        imgs = sorted(os.listdir(imdir))
#        if args.shuffle:
#            info('Shuffling images')
#            random.shuffle(imgs)
#
#        for im in imgs:
#            if not im.endswith('.jpg'): continue
#            # info('{}'.format(im))
#            wktpath = pjoin(wktdir, os.path.splitext(im)[0]+'.wkt')
#            masksumpath = pjoin(masksumdir, os.path.splitext(im)[0]+'.txt')
#
#            if os.path.exists(wktpath): continue
#
#            t0 = time.time()
#            impath = os.path.join(imdir, im)
#            try:
#                original_im = Image.open(impath).convert('RGB')
#            except:
#                continue
#
#            segmap = MODEL.run(np.array(original_im))
#            segmap[segmap !=1] = 0 # Just tag!
#
#            masksum = np.sum(segmap[:])
#            with open(masksumpath, 'w') as fh:
#                fh.write(str(masksum))
#
#            polys = get_contours(segmap)
#            dump_contours_to_wkt(polys, wktpath)
#
#            crop_masks(original_im, im, polys, cropdir)
#
#
###########################################################
