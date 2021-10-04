#!/usr/bin/env python3
"""
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme

import torch, torchvision
import mmcv
import mmseg
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from subprocess import Popen

##########################################################
def save_result_pyplot(model,
                       img,
                       result,
                       ax,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    info('Pytorch:', torch.__version__, torch.cuda.is_available())
    info('MMseg:', mmseg.__version__)

    modeluri = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    checkp_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'

    if not os.path.exists(checkp_path):
        cmd = 'wget {} -P checkpoints'.format(modeluri)
        p = Popen(cmd, shell=True)
        stdout, stderr = p.communicate()
        print(stdout, stderr)

    model = init_segmentor(config_file, checkp_path, device='cuda:0')

    img = 'demo.png'
    result = inference_segmentor(model, img)

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    save_result_pyplot(model, img, result, ax, get_palette('cityscapes'))
    outpath = pjoin(outdir, 'demo.png')
    plt.savefig(outpath)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
