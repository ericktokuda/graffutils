#!/usr/bin/env python3
"""
"""

import argparse
import time, datetime
import os
import os.path as osp
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from subprocess import Popen
from PIL import Image
from myutils import info, create_readme

import torch, torchvision
import mmcv
from mmcv import Config
import mmseg
from mmseg.apis import set_random_seed
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

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
def create_config(data_root, img_dir, ann_dir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.num_classes = 8
    cfg.model.auxiliary_head.num_classes = 8

    cfg.dataset_type = 'StandfordBackgroundDataset'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu=8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(320, 240),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'splits/val.txt'

    cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

    cfg.work_dir = './work_dirs/tutorial'

    cfg.runner.max_iters = 200
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    print(f'Config:\n{cfg.pretty_text}')
    return cfg

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    info('Pytorch:', torch.__version__, torch.cuda.is_available())
    info('MMseg:', mmseg.__version__)

    """
    To train on a customized dataset, the following steps are neccessary.
    1. Add a new dataset class.
    2. Create a config file accordingly.
    3. Perform training and evaluation.

    Classes: sky, tree, road, grass, water, building, mountain, and foreground object.
    """

    if not os.path.exists('./iccv09Data'):
        info('Downloading ...')
        cmd = 'wget http://dags.stanford.edu/data/iccv09Data.tar.gz -O standford_background.tar.gz'
        p = Popen(cmd, shell=True)
        stdout, stderr = p.communicate(); print(stdout, stderr)

        cmd = 'tar xf standford_background.tar.gz'
        p = Popen(cmd, shell=True)
        stdout, stderr = p.communicate(); print(stdout, stderr)

    info('Extracting ...')
    img = mmcv.imread('iccv09Data/images/6000124.jpg')
    plt.figure(figsize=(8, 6))
    plt.imshow(mmcv.bgr2rgb(img))
    plt.savefig(pjoin(outdir, '6000124.png'))

    info('Converting annotation ...')
    data_root = 'iccv09Data'
    img_dir = 'images'
    ann_dir = 'labels'
    classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
    palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

    for file in mmcv.scandir(pjoin(data_root, ann_dir), suffix='.regions.txt'):
        seg_map = np.loadtxt(pjoin(data_root, ann_dir, file)).astype(np.uint8)
        seg_img = Image.fromarray(seg_map).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(pjoin(data_root, ann_dir, file.replace('.regions.txt', '.png')))

    info('Exploring one image of the dataset (6000124.png) ...')
    img = Image.open('iccv09Data/labels/6000124.png')
    plt.figure(figsize=(8, 6))
    im = plt.imshow(np.array(img.convert('RGB')))
    patches = [mpatches.Patch(color=np.array(palette[i])/255.,
                              label=classes[i]) for i in range(8)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
               fontsize='large')
    plt.savefig(pjoin(outdir, '6000124_label.png'))

    info('Splitting train-val...')
    split_dir = 'splits'
    mmcv.mkdir_or_exist(pjoin(data_root, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
        pjoin(data_root, ann_dir), suffix='.png')]
    with open(pjoin(data_root, split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
        train_length = int(len(filename_list)*4/5)
        f.writelines(line + '\n' for line in filename_list[:train_length])
    with open(pjoin(data_root, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
        f.writelines(line + '\n' for line in filename_list[train_length:])

    @DATASETS.register_module()
    class StandfordBackgroundDataset(CustomDataset):
        CLASSES = classes
        PALETTE = palette
        def __init__(self, split, **kwargs):
            super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                             split=split, **kwargs)
            assert osp.exists(self.img_dir) and self.split is not None

    cfg = create_config(data_root, img_dir, ann_dir)

    info('Training...')
    datasets = [build_dataset(cfg.data.train)]

    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())

    info('Evaluating...')
    img = mmcv.imread('iccv09Data/images/6000124.jpg')

    model.cfg = cfg
    result = inference_segmentor(model, img)
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    save_result_pyplot(model, img, result, ax, palette)

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
