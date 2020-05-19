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
from mpl_toolkits.mplot3d import axes3d

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

#############################################################
def plot_contours(f, x, y, xx, yy, xrange, yrange, outdir):
    info(inspect.stack()[0][3] + '()')

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
def plot_densities(df, xx, yy, outdir):
    """Plot the densities in the grid @xx, @yy
    """
    info(inspect.stack()[0][3] + '()')

    annotators = np.unique(df.annotator)
    labels = np.unique(df.label)

    nrows = len(annotators);  ncols = len(labels)
    figscale = 4
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))
    for i, anno in enumerate(annotators):
        axs[i, 0].set_ylabel('Annotator {}'.format(i))
        for j, l in enumerate(labels):
            if i == 0: axs[i, j].set_title('Type {}'.format(l))
            filtered = df[(df.annotator == anno) & (df.label == int(l))]
            f = compute_pdf_over_grid(filtered.x, filtered.y, xx, yy)
            axs[i, j].scatter(xx, yy, c=f)

    # gdf = gpd.read_file(shppath)
    # shapefile = gdf.geometry.values[0]
    # xs, ys = shapefile.exterior.xy
    # ax[0, 0].plot(xs, ys, c='dimgray')
    plt.savefig(pjoin(outdir, 'density.png'))

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
    yrange = [np.min(y) - marginy - .1, np.max(y) + marginy]
    return np.mgrid[xrange[0]:xrange[1]:100j, yrange[0]:yrange[1]:100j]

#############################################################
def process(clusterlabelspath, outdir):
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(clusterlabelspath)
    xx, yy = create_meshgrid(df.x, df.y, relmargin=.1)

    # f = compute_pdf_over_grid(df.x, df.y, xx, yy)
    # plt.scatter(df.x, df.y); plt.savefig(pjoin(outdir, 'points.pdf'))
    # plot_hist2d(df.x, df.y, outdir)
    # plot_surface(f, df.x, df.y, xx, yy, outdir)
    # plot_contours(f, df.x, df.y, xx, yy, xrange, yrange, outdir)
    # plot_surface(f, df.x, df.y, xx, yy, outdir)
    # plot_wireframe(f, df.x, df.y, xx, yy, outdir)

    plot_densities(df, xx, yy, outdir)
    
##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clusterlabels', required=True,
            help='Path to the csv containing the cluster and labels for each location')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    process(args.clusterlabels, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
