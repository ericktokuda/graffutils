#!/usr/bin/env python3
""" DeepLab evaluation
Tensorflow 1.x is required (tested on tensorflow==1.15).
"""

import sys
import cv2
import shapely
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
import os
import random
import inspect
import imageio

import argparse
import logging
from os.path import join as pjoin

from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

import matplotlib
from matplotlib import gridspec, cm
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
import time

from myutils import info

#########################################################
class DeepLabModel(object):
  """Deeplab model for prediction"""
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 640

  def __init__(self, frozenpath):
    """Loads pretrained deeplab model"""
    self.graph = tf.Graph()
    graph_def = tf.GraphDef.FromString(open(frozenpath, 'rb').read())

    with self.graph.as_default():
        tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def predict(self, image):
    """Prediction on @image and return the np.array mask """
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [image]})
    return batch_seg_map[0]

##########################################################
def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.  """

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

##########################################################
def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """

  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

##########################################################
def get_contours(mask_):
    epsilon = 2
    aux, _ = cv2.findContours(mask_.astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for cnt in aux:
        aux2 = cv2.approxPolyDP(cnt, epsilon, True)
        if len(aux2) < 3: continue
        # aux3 = get_2dpoints_from_cv2_struct(aux2)
        aux3 = aux2
        contours.append(aux3)
    return contours

##########################################################
def dump_contours_to_wkt(polys, wktpath):
    if len(polys) == 0:
        Path(wktpath).touch()
        return

    shapelyinput = []

    for poly in polys:
        points = [ list(p[0]) for p in poly ]
        shapelyinput.append([points,[]])

    shapelycontours = MultiPolygon(shapelyinput)
    with open(wktpath, 'w') as fh:
        fh.write(shapelycontours.wkt)

##########################################################
def crop_masks(im, imfilename, polys, cropdir):
    if len(polys) == 0: return

    shapelyinput = []

    for i, poly in enumerate(polys):
        croppath = pjoin(cropdir,
                        '{}_{:02d}.jpg'. \
                        format(os.path.splitext(imfilename)[0], i))
        coords = np.squeeze(poly)
        xmin, ymin = np.min(coords, 0)
        xmax, ymax = np.max(coords, 0)

        if xmax - xmin < 10 or ymax - ymin < 10 : continue # Too small

        cropped = im.crop((xmin, ymin, xmax, ymax))
        cropped.save(croppath)

##########################################################
def run_visualization(im, immask, labels, outpath):
    """Inferences DeepLab model and visualizes result."""

    figsize = (12, 6)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].imshow(im)
    imgmean = np.mean(im, axis=2)
    axs[1].imshow(imgmean, alpha=.8, cmap='gray')

    clrs = ['#ffffff','#377eb8','#4daf4a','#984ea3','#ff7f00']
    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',
            clrs, len(labels))
    implot = axs[1].imshow(immask, cmap=mycmap, alpha=.7, vmin=0, vmax=len(labels)-1)
    cbar = fig.colorbar(implot, ticks=range(len(labels)), shrink=.75)
    cbar.ax.set_ylim([0, len(labels)])
    cbar.ax.set_yticklabels(labels)

    axs[0].set_xticks([]); axs[0].set_yticks([]);
    axs[1].set_xticks([]); axs[1].set_yticks([]);
    plt.tight_layout()
    plt.savefig(vispath)

##########################################################
def predict_all(modelpath, dirpaths, outdir):
    """Predict using frozen @modelpath all images in @imdir and outputs
    to @outdir
    """
    info(inspect.stack()[0][3] + '()')

    wktdir = pjoin(outdir, 'wkt')
    if not os.path.exists(wktdir): os.mkdir(wktdir)

    mskdir = pjoin(outdir, 'masksum')
    if not os.path.exists(mskdir): os.mkdir(mskdir)

    labels = np.array(['background', 'tag', 'frame', 'sign'])

    MODEL = DeepLabModel(modelpath)
    info('Model loaded successfully!')

    for dir_ in dirpaths:
        imdir = pjoin(dir_, 'img/')
        info('{}'.format(dir_))

        lastpath = os.path.basename(os.path.normpath(dir_))
        wktdir = pjoin(outdir, 'wkt', lastpath)
        os.makedirs(wktdir, exist_ok=True)

        masksumdir = pjoin(outdir, 'masksum', lastpath)
        os.makedirs(masksumdir, exist_ok=True)

        cropdir = pjoin(outdir, 'crop', lastpath)
        os.makedirs(cropdir, exist_ok=True)

        files = sorted(os.listdir(imdir))
        for f in files:
            info('file:{}'.format(f))
            if not f.endswith('.jpg'): continue
            suff = os.path.splitext(f)[0]
            imgpath = os.path.join(imgdir, f)

            try: img = imageio.imread(imgpath)
            except: continue

            mask = model.predict(img)

            # vis
            vispath = pjoin(outdir, os.path.basename(imgpath))
            run_visualization(img, mask, labels, vispath)
            continue

            # info('{}'.format(im))
            wktpath = pjoin(wktdir, suff +'.wkt')
            masksumpath = pjoin(masksumdir, suff +'.txt')

            if os.path.exists(wktpath): continue

            mask[mask !=1] = 0 # Just tag!

            masksum = np.sum(mask[:]) # dump masksum
            with open(masksumpath, 'w') as fh:
                fh.write(str(masksum))

            polys = get_contours(segmap)
            dump_contours_to_wkt(polys, wktpath) # dump contours

            crop_masks(img, f, polys, cropdir)

##########################################################
def analyze_deeplab_log(logpath):
    """Parse deeplab log in an attempt to find the best iou
    It is a bit crypt here.
    """
    info(inspect.stack()[0][3] + '()')
    if not os.path.exists(outdir): os.mkdir(outdir)
    fh = open(logpath, 'r')
    res = []
    while True:
       aux = fh.readline()
       ckpt = int(aux.replace('model.ckpt-', '').replace('.meta', ''))
       print(ckpt)
       aux = fh.readline()
       idx = aux.find('class_0')
       aux = aux[idx+12:]
       idx = aux.find(']')
       aux = aux[:idx]
       iou0 = float(aux)

       aux = fh.readline()
       idx = aux.find('class_1')
       aux = aux[idx+12:]
       idx = aux.find(']')
       aux = aux[:idx]
       iou1 = float(aux)

       res.append([ckpt, iou0, iou1, (iou0+iou1)/2])

       if ckpt == 9730: break

    fh.close()

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozenpath', required=True, help='Frozen model path')
    parser.add_argument('--imdir', required=True, help='Folder of the images')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.mkdir(args.outdir)
    # predict_all(args.frozenpath, args.imdir, args.outdir)
    predict_all(args.frozenpath, [args.imdir], args.outdir)

##########################################################
if __name__ == "__main__":
    main()

