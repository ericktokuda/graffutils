#!/usr/bin/env python3
""" DeepLab evaluation
"""

import sys
import cv2
import shapely
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
import os
import random

import argparse
import logging
from os.path import join as pjoin

from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
import time

from src.utils import info

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

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
        tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def predict(self, image):
    """Prediction on @image and return the np.array mask
    """
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [image]})
    return batch_seg_map[0]

##########################################################
def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
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
def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), labels[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()



# def get_2dpoints_from_cv2_struct(cv2points):
    # points = []
    # for cv2point in cv2points:
        # points.append(cv2point[0])
    # return points

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

    # print('wktpath:{}'.format(wktpath))
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
def run_visualization(impath):
  """Inferences DeepLab model and visualizes result."""

  original_im = Image.open(impath)
  resized_im, seg_map = MODEL.run(original_im)

  vis_segmentation(resized_im, seg_map)
##########################################################
def predict_all(modelpath, imdir, outdir):
    """Predict using frozen @modelpath all images in @imdir and outputs
    to @outdir
    """
    info(inspect.stack()[0][3] + '()')

    labels = np.asarray('background tag frame sign'.split(' '))

    FULL_LABEL_MAP = np.arange(len(labels)).reshape(len(labels), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    MODEL = DeepLabModel(args.frozenpath)
    info('Model loaded successfully!')

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(pjoin(args.outdir, 'wkt'), exist_ok=True)
    os.makedirs(pjoin(args.outdir, 'masksum'), exist_ok=True)

    dirpaths = open(args.dirslist).read().splitlines()
    print('dirpaths:{}'.format(dirpaths))

    for dir_ in dirpaths:
        imdir = pjoin(dir_, 'img/')
        info('{}'.format(dir_))

        lastpath = os.path.basename(os.path.normpath(dir_))
        wktdir = pjoin(args.outdir, 'wkt', lastpath)
        os.makedirs(wktdir, exist_ok=True)

        masksumdir = pjoin(args.outdir, 'masksum', lastpath)
        os.makedirs(masksumdir, exist_ok=True)

        cropdir = pjoin(args.outdir, 'crop', lastpath)
        os.makedirs(cropdir, exist_ok=True)

        imgs = sorted(os.listdir(imdir))
        for im in imgs:
            if not im.endswith('.jpg'): continue
            # info('{}'.format(im))
            wktpath = pjoin(wktdir, os.path.splitext(im)[0]+'.wkt')
            masksumpath = pjoin(masksumdir, os.path.splitext(im)[0]+'.txt')

            if os.path.exists(wktpath): continue

            t0 = time.time()
            impath = os.path.join(imdir, im)
            try:
                original_im = Image.open(impath).convert('RGB')
            except:
                continue

            segmap = MODEL.run(np.array(original_im))
            segmap[segmap !=1] = 0 # Just tag!

            masksum = np.sum(segmap[:])
            with open(masksumpath, 'w') as fh:
                fh.write(str(masksum))

            polys = get_contours(segmap)
            dump_contours_to_wkt(polys, wktpath)

            crop_masks(original_im, im, polys, cropdir)

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

    if not os.path.exists(outdir): os.mkdir(outdir)
    predict_all(args.frozenpath, imdir, args.outdir)

##########################################################
if __name__ == "__main__":
    main()

