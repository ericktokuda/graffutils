#!/usr/bin/env python
"""Perform kernel density estimation per type
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
from scipy import stats
import pandas as pd
import geopandas as geopd
from mpl_toolkits.mplot3d import axes3d

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

#############################################################
def plot_contours(f, x, y, xx, yy, outdir):
    info(inspect.stack()[0][3] + '()')

    xrange = [np.min(xx), np.max(xx)]
    yrange = [np.min(yy), np.max(yy)]
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm',
            extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    plt.title('2D Gaussian Kernel density estimation')
    plt.savefig(pjoin(outdir, 'contours.pdf'))

#############################################################
def plot_surface(f, x, y, xx, yy, outdir):
    info(inspect.stack()[0][3] + '()')
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1,
            cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(60, 35)
    plt.savefig(pjoin(outdir, 'surfaceplot.pdf'))

#############################################################
def plot_wireframe(f, x, y, xx, yy, outdir):
    info(inspect.stack()[0][3] + '()')
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    w = ax.plot_wireframe(xx, yy, f)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Wireframe plot of Gaussian 2D KDE');
    plt.savefig(pjoin(outdir, 'wireframe.pdf'))

#############################################################
def plot_hist2d(x, y, outdir):
    info(inspect.stack()[0][3] + '()')
    # h = plt.hist2d(x, y, bins=1000, density=True)
    h = plt.hist2d(x, y, density=True)
    plt.colorbar(h[3])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Frequency histogram')
    plt.savefig(pjoin(outdir, 'hist2d.pdf'))

##########################################################
def plot_density_real(df, xx, yy, mapx, mapy, outdir):
    """Plot the densities in the grid @xx, @yy
    """
    info(inspect.stack()[0][3] + '()')

    annotators = np.unique(df.annotator)
    labels = np.unique(df.label)

    nrows = len(annotators);  ncols = len(labels)
    figscale = 4
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))

    pdfvals = np.ndarray((len(annotators), len(labels), xx.shape[0], xx.shape[1]))
    for i, anno in enumerate(annotators):
        for j, l in enumerate(labels):
            filtered = df[(df.annotator == anno) & (df.label == int(l))]
            pdfvals[i, j, :, :] = compute_pdf_over_grid(filtered.x, filtered.y, xx, yy)

    for i, anno in enumerate(annotators):
        axs[i, 0].set_ylabel('Annotator {}'.format(i))
        # if i == 0: axs[i, 0].set_title('Mean pdf')
        for j, l in enumerate(labels):
            if i == 0: axs[i, j].set_title('Type {}'.format(l))
            vals = pdfvals[i][j]
            axs[i, j].plot(mapx, mapy, c='dimgray')
            im = axs[i, j].scatter(xx, yy, c=vals)
            # fig.colorbar(im, ax=axs[i, j])

    plt.tight_layout(2)
    plt.savefig(pjoin(outdir, 'density_real.png'))

##########################################################
def plot_density_diff_to_mean(df, xx, yy, mapx, mapy, outdir):
    """Plot the densities in the grid @xx, @yy
    """
    info(inspect.stack()[0][3] + '()')

    annotators = np.unique(df.annotator)
    labels = np.unique(df.label)

    nrows = len(annotators);  ncols = len(labels) + 1
    figscale = 4
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))

    pdfvals = np.ndarray((len(annotators), len(labels), xx.shape[0], xx.shape[1]))
    for i, anno in enumerate(annotators):
        for j, l in enumerate(labels):
            filtered = df[(df.annotator == anno) & (df.label == int(l))]
            pdfvals[i, j, :, :] = compute_pdf_over_grid(filtered.x, filtered.y, xx, yy)

    for i, anno in enumerate(annotators):
        axs[i, 0].set_ylabel('Annotator {}'.format(i))
        meanpdf = np.mean(pdfvals[i], axis=0)
        im = axs[i, 0].scatter(xx, yy, c=meanpdf)
        # fig.colorbar(im, ax=axs[i, 0])
        if i == 0: axs[i, 0].set_title('Mean pdf')
        for j, l in enumerate(labels):
            jj = j + 1
            if i == 0: axs[i, jj].set_title('Type {}'.format(l))
            vals = pdfvals[i][j] - meanpdf
            axs[i, jj].plot(mapx, mapy, c='dimgray')
            im = axs[i, jj].scatter(xx, yy, c=vals)
            # fig.colorbar(im, ax=axs[i, jj])

    plt.tight_layout(2)
    plt.savefig(pjoin(outdir, 'density_difftomean.png'))

##########################################################
def plot_density_pairwise_diff(df, xx, yy, mapx, mapy, outdir):
    """Plot all the combinations of the differences on the densities
    in the grid @xx, @yy
    """
    info(inspect.stack()[0][3] + '()')

    labels = np.unique(df.label)
    from itertools import combinations
    combs = list(combinations(list(range(len(labels))), 2))

    nrows = 1;  ncols = len(combs)
    figscale = 4
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))


    pdfvals = np.ndarray((len(labels), xx.shape[0], xx.shape[1]))
    for j, l in enumerate(labels):
        filtered = df[(df.label == int(l))]
        pdfvals[j, :, :] = compute_pdf_over_grid(filtered.x, filtered.y, xx, yy)

    i = 0
    for j, comb in enumerate(combs):
        vals = pdfvals[comb[1]] - pdfvals[comb[0]]
        axs[i, j].plot(mapx, mapy, c='dimgray')
        im = axs[i, j].scatter(xx, yy, c=vals)
        # fig.colorbar(im, ax=axs[i, j])
        axs[i, j].set_title('Type{} - Type{}'.format(comb[1], comb[0]))

    plt.tight_layout(2)
    plt.savefig(pjoin(outdir, 'density_pairwisediff.png'))

##########################################################
def compute_pdf_over_grid(x, y, xx, yy):
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    return np.reshape(kernel(positions).T, xx.shape)

##########################################################
def create_meshgrid(x, y, relmargin=.1):
    """Create a meshgrid for @x and @y with margins
    Receives  and returns a ret
    """
    info(inspect.stack()[0][3] + '()')
    marginx = (max(x) - min(x)) * relmargin
    marginy = (max(y) - min(y)) * relmargin

    xrange = [np.min(x) - marginx, np.max(x) + marginx]
    # yrange = [np.min(y) - marginy, np.max(y) + marginy] 
    yrange = [np.min(y) - marginy - .15, np.max(y) + marginy] 
    return np.mgrid[xrange[0]:xrange[1]:100j, yrange[0]:yrange[1]:100j]

#############################################################
def get_shp_points(shppath):
    """Get points from @shppath and returns list of points, x and y
    """
    info(inspect.stack()[0][3] + '()')
    geodf = geopd.read_file(shppath)
    shapefile = geodf.geometry.values[0]
    return shapefile.exterior.xy

#############################################################
def process(clusterlabelspath, shppath, outdir):
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(clusterlabelspath)
    xx, yy = create_meshgrid(df.x, df.y, relmargin=.1)

    # f = compute_pdf_over_grid(df.x, df.y, xx, yy)
    # plt.scatter(df.x, df.y); plt.savefig(pjoin(outdir, 'points.pdf'))
    # plot_hist2d(df.x, df.y, outdir)
    # plot_surface(f, df.x, df.y, xx, yy, outdir)
    # plot_contours(f, df.x, df.y, xx, yy, outdir)
    # plot_surface(f, df.x, df.y, xx, yy, outdir)
    # plot_wireframe(f, df.x, df.y, xx, yy, outdir)

    mapx, mapy = get_shp_points(shppath)
    plot_density_real(df, xx, yy, mapx, mapy, outdir)
    plot_density_diff_to_mean(df, xx, yy, mapx, mapy, outdir)
    plot_density_pairwise_diff(df, xx, yy, mapx, mapy, outdir)
    
##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clusterlabels', required=True,
            help='Path to the csv containing the cluster and labels for each location')
    parser.add_argument('--shppath', required=True, help='Path to the SHP dir')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    plt.rcParams['image.cmap'] = 'Blues'
    process(args.clusterlabels, args.shppath, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
