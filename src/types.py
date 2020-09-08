#!/usr/bin/env python3
"""Analyze labels generated by mannual classification of the crops
"""

import argparse
import time
from os.path import join as pjoin
import os
import inspect

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.collections as mc
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from scipy.spatial import cKDTree
import scipy.stats as stats
import igraph
import geopandas as geopd
from geopandas import GeoSeries
import matplotlib_venn
from src.utils import info, export_individual_axis, hex2rgb
# plt.style.use('seaborn')
from myutils import graph, geo
import scipy
import scipy.spatial
from numba import jit
import pickle
from multiprocessing import Pool
from functools import partial
import shapely; from shapely.geometry import Point;

palettehex = ['#8dd3c7','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69']
palette = hex2rgb(palettehex, normalized=True, alpha=1.0)
palettehex2 = ['#1b9e77','#d95f02','#7570b3','#e7298a']
palette2 = hex2rgb(palettehex2, normalized=True, alpha=1.0)
palettehex3 = ['#e41a1c','#377eb8','#e6e600','#984ea3','#ff7f00','#4daf4a','#a65628','#f781bf','#999999']
palette3 = hex2rgb(palettehex3, normalized=True, alpha=.7)
palette3[5,3] = 1.0

R = 6371000
##########################################################
def plot_types(infomapout, shppath, clulabelspath, outdir):
    np.random.seed(0)
    df = pd.read_csv(clulabelspath, index_col='id')
    totalrows = len(df)

    fig, ax = plt.subplots(1, 1, figsize=(15/2, 10), squeeze=False) # Plot contour
    geodf = geopd.read_file(shppath)
    shapefile = geodf.geometry.values[0]
    
    xs, ys = shapefile.exterior.xy
    ax[0, 0].plot(xs, ys, c='dimgray')

    clusters = np.unique(df.cluster)
    clusters_str = ['C{}'.format(cl) for cl in clusters]
    nclusters = len(clusters)
    labels = np.unique(df.label)
    nlabels = len(labels)

    markers = ['$A$', '$B$', '$C$']
    ss = [30, 35, 35]
    edgecolours = ['#993333', '#339933', '#3366ff']
    visual = [ dict(marker=m, s=s, edgecolors=e) for m,s,e in \
              zip(['o', 'o', 'o'], ss, edgecolours)]

    for i, l in enumerate(labels):
        data = df[df.label == l]
        ax[0, 0].scatter(data.x, data.y, c=edgecolours[i],
                         label='Type ' + markers[i],
                         alpha=0.6,
                         # linewidths=0.2,
                         # edgecolor=(0.3, 0.3, 0.3, 1),
                         **(visual[i]))
    
    fig.patch.set_visible(False)
    ax[0, 0].axis('off')

    ax[0, 0].legend(loc=(0.6, 0.12), title='Graffiti types',
                    fontsize='xx-large', title_fontsize='xx-large')

    extent = ax[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(pjoin(outdir, 'map_types.pdf'), bbox_inches=extent.expanded(1.0, 1.0))

##########################################################
def plot_counts_normalized(clulabelspath, cluareaspath, outdir):
    df = pd.read_csv(clulabelspath, index_col='id')
    totalrows = len(df)

    clusters = np.unique(df.cluster)
    clusters_str = ['C{}'.format(cl) for cl in clusters]
    nclusters = len(clusters)
    labels = np.unique(df.label)
    nlabels = len(labels)
    # labels_str = [str(l) for l in labels]
    plotsize = 5
    alpha = 0.6
    palette = np.array([
        [27.0,158,119],
        [217,95,2],
        [117,112,179],
        [231,41,138],
        [102,166,30],
        [230,171,2],
    ])
    palette /= 255.0
    colours = np.zeros((palette.shape[0], 4), dtype=float)
    colours[:, :3] = palette
    colours[:, -1] = alpha

    counts = np.zeros(nclusters, dtype=int)
    countsnorm = np.zeros(nclusters)
    areas = pd.read_csv(cluareaspath)

    for i, cl in enumerate(clusters):
        data = df[df.cluster == cl]
        counts[i] = len(data)
        points = data[['x', 'y']].values

        countsnorm[i] = counts[i] / areas.iloc[i]

    fig, ax = plt.subplots(1, 1, figsize=(2*plotsize, plotsize),
                           squeeze=False)
    yfactor = 1
    ax[0, 0].bar(clusters_str, countsnorm / yfactor, color=colours)
    ax[0, 0].set_ylabel('Normalized count of graffitis')
    ax[0, 0].set_xlabel('Community')
    i = 0
    for spine in ax[0, i].spines.values():
        spine.set_edgecolor('dimgray')
    ax[0, i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[0, i].spines['top'].set_visible(False)
    ax[0, i].spines['right'].set_visible(False)
    ax[0, i].yaxis.grid(True, alpha=0.4)
    ax[0, i].set_axisbelow(True)
    # ax[0, i].spines['left'].set_visible(False)

    plt.savefig(pjoin(outdir, 'countsnormalized.pdf'))

##########################################################
def count_labels_per_region(df, clusters, labels, cluids):
    """Count number of labels per region """
    nlabels = len(labels)
    nclusters = len(clusters)

    counts = np.ones((nclusters, nlabels), dtype=float)
    for i in range(nclusters):
        labels_reg, counts_reg = np.unique(df[df.index.isin(cluids[i])].label,
                                           return_counts=True)
        for j in range(nlabels):
            lab = labels[j]
            if not lab in labels_reg: continue
            ind = np.where(labels_reg == lab)
            counts[i, j] = counts_reg[ind]
    return counts

#############################################################
def shuffle_labels(labelspath, outdir):
    """Shuffle labels from @labelspath and compute metrics """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(labelspath, index_col='id')

    nrealizations = 10
    labels = np.unique(df.label)
    clusters = np.unique(df.cluster)
    nlabels = len(labels)
    nclusters = len(clusters)
    info('nrealizations:{}, nclusters:{}'.format(nrealizations, nclusters))

    cluids = {}
    for i in range(nclusters):
        # cluids[i] = df[df.cluster == clusters[i]].index
        cluids[i] = np.where(df.cluster.values == clusters[i])[0]

    counts_orig = count_labels_per_region(df, clusters, labels, cluids)
    counts_perm = count_shuffled_labels_per_region(df, clusters, labels,
            cluids, nrealizations)

    plot_shuffle_distrib_and_orig(counts_orig, counts_perm, nclusters,
            nlabels, outdir)

##########################################################
def compile_lists(listsdir, labelspath):
    """Compile lists (.lst) in @listdir """
    info(inspect.stack()[0][3] + '()')
    files = sorted(os.listdir(listsdir))

    cols = 'img,x,y,label'.split(',')
    data = []
    for f in files:
        if not f.endswith('.lst'): continue
        label = int(f.replace('.lst', '').split('_')[1])
        lines = open(pjoin(listsdir, f)).read().strip().splitlines()
        for l in lines:
            id = l.replace('.jpg', '')
            _, y, x, heading = id.split('_')
            data.append([l, x, y, label])
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(labelspath, index_label='id',)

#############################################################
def compile_labels(annotdir, labelspath):
    """Summarize @annotdir csv annotations in .txt format and output
    summary to @labelspath """
    info(inspect.stack()[0][3] + '()')

    if os.path.exists(labelspath):
        info('Loading {}'.format(labelspath))
        return pd.read_csv(labelspath)

    files = sorted(os.listdir(annotdir))

    labels = '1 2 3'.split(' ')
    info('Using labels:{}'.format(labels))

    cols = 'img,x,y,label'.split(',')
    data = []
    for f in files:
        if not f.endswith('.txt'): continue
        filepath = pjoin(annotdir, f)
        _, y, x, heading = os.path.split(filepath)[-1].replace('.txt', '').split('_')
        labels_ = open(filepath).read().strip().split(',')

        for l in labels_: # each label in the file correspond to a new row
            img = f.replace('.txt', '.jpg')
            data.append([img, x, y, l])

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(labelspath, index_label='id',)
    return df

##########################################################
def parse_infomap_results(graphml, infomapout, labelsdf, annotator,
        labelsclupath, vcoordspath):
    """Find enclosing community given by @infomapout of each node in @graphml """
    info(inspect.stack()[0][3] + '()')

    if os.path.exists(labelsclupath):
        return pd.read_csv(labelsclupath), pd.read_csv(vcoordspath).values

    g = graph.simplify_graphml(graphml, directed=True, simplify=True)
    vcoords = np.array([g.vs['x'], g.vs['y']]).T
    g = graph.simplify_graphml(graphml, directed=True, simplify=True)
    cludf = pd.read_csv(infomapout, sep=' ', skiprows=[0, 1],
                     names=['id', 'cluster','flow']) # load graph clusters
    cludf = cludf.sort_values(by=['id'], inplace=False)

    coords_objs = np.zeros((len(labelsdf), 2))
    
    i = 0
    for _, row in labelsdf.iterrows():
        coords_objs[i, 0] = row.x
        coords_objs[i, 1] = row.y
        i += 1

    kdtree = cKDTree(vcoords)
    dists, inds = kdtree.query(coords_objs)
    labelsdf['cluster'] = np.array(cludf.cluster.tolist())[inds]
    labelsdf['annotator'] = annotator
    labelsdf.to_csv(labelsclupath, index=False, float_format='%.08f')
    pd.DataFrame(vcoords, columns=['x', 'y']).to_csv(vcoordspath, index=False)
    
    return labelsdf, vcoords

##########################################################
def convert_csv_to_annotdir(labelsclu, annotator, outdir):
    """Convert dataframe in @labelsclu from @annotator to txt format in @outdir """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(labelsclu)
    labels = np.unique(df.label)
    labeldir = pjoin(outdir, 'annot')
    if not os.path.isdir(labeldir): os.mkdir(labeldir)

    filtered = df[(df.annotator == annotator)]
    imgs = np.unique(filtered.img)

    for im in imgs:
        aux = filtered[filtered.img == im]
        mylabels = sorted(np.array(list(set(aux.label))).astype(str))
        if '1' in mylabels and len(mylabels) == 1: print(im)
        mylabelsstr = ','.join(mylabels)
        annotpath = pjoin(labeldir, im.replace('.jpg', '.txt'))
        open(pjoin(labeldir, annotpath), 'w').write(mylabelsstr)

##########################################################
def plot_venn(labelsclupath, outdir):
    """Plot venn diagram """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(labelsclupath)
    labels = sorted(np.unique(df.label))

    img2id = {}
    for i, img in enumerate(sorted(np.unique(df.img))):
        img2id[img] = i

    subsets = []
    for l in labels:
        aux = df[df.label == l].img.tolist()
        partition = [img2id[k] for k in aux]
        subsets.append(set(partition))

    # edgecolours = ['#993333', '#339933', '#3366ff']
    plt.figure(figsize=(4,3))
    # cl = [[167/255, 167/255, 167/255, 1.0]] * 3
    cl = [[114/255, 105/255, 121/255, 1.0]] * 3
    
    matplotlib_venn.venn3(subsets,
            set_labels = ('TypeA', 'TypeB', 'TypeC'),
            # set_colors=palettehex3[6:],
            set_colors=cl,
            alpha=.7,
            )
    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'counts_venn.pdf'))


#########################################################
def plot_stacked_bar_types(results, nclusters, nlabels,
        colours, outdir):
    """Plot each row of result as a horiz stacked bar plot"""
    fig, ax = plt.subplots(figsize=(5, 4))
    n, m = results.shape

    letters = 'ABCDE'
    rownames = [ 'C{}'.format(i+1) for i in range(nclusters)]
    colnames = [ 'Type {}'.format(letters[i]) for i in range(nlabels)]

    prev = np.zeros(n)
    ps = []
    for j in range(m):
        p = ax.barh(range(n), results[:, j], left=prev, height=.6,
                color=colours[6+j])
        prev += results[:, j]
        ps.append(p)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(rownames)
    ax.grid(False)
    ax.set_xlabel('Ratio')
    xticks = np.array([0, .2, .4, .6, .8, 1.0])
    for spine in ax.spines.values():
        spine.set_edgecolor('dimgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Community')
    ax.legend(ps, colnames, ncol=len(colnames), bbox_to_anchor=(0.08, 1),
          loc='lower left') #, fontsize='small')
    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'ratio_bars.pdf'))

##########################################################
def get_ratios_by_community(labelsdf, clusters, normalized=True):
    """Get ratio for each community """
    info(inspect.stack()[0][3] + '()')
    results = np.zeros((len(clusters), len(np.unique(labelsdf.label))))
    for i, cl in enumerate(clusters):
        results[i, :] = labelsdf[labelsdf.cluster == cl]. \
                groupby('label').sum().cluster.values
    if normalized: results = results / np.sum(results, axis=1).reshape(-1, 1)
    return results

##########################################################
########################################################## KDE
##########################################################

def create_meshgrid(x, y, nx=100, ny=100, relmargin=.1):
    """Create a meshgrid around @x and @y with @nx, @ny tiles and relative
    margins @relmargins"""

    marginx = (max(x) - min(x)) * relmargin
    marginy = (max(y) - min(y)) * relmargin
    xrange = [np.min(x) - marginx, np.max(x) + marginx]
    yrange = [np.min(y) - marginy - .15, np.max(y) + marginy] 
    dx = (xrange[1] - xrange[0]) / nx
    dy = (yrange[1] - yrange[0]) / ny
    xx, yy = np.mgrid[xrange[0]:xrange[1]:(nx*1j), yrange[0]:yrange[1]:(ny*1j)]
    return xx, yy, dx, dy

##########################################################
def compute_pdf_over_grid(x, y, xx, yy, kerbw):
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values, bw_method=kerbw)
    return np.reshape(kernel(positions).T, xx.shape), kernel.factor

#############################################################
def plot_contours(f, x, y, xx, yy, outdir):
    info(inspect.stack()[0][3] + '()')

    xrange = [np.min(xx), np.max(xx)]
    yrange = [np.min(yy), np.max(yy)]
    fig, ax = plt.subplots(figsize=(8,8))
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
    h = plt.hist2d(x.values, y.values, density=True)
    plt.colorbar(h[3])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Frequency histogram')
    plt.savefig(pjoin(outdir, 'hist2d.png'))

##########################################################
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

##########################################################
def get_points_inside_region(df, c0, radius):
    """Get points from @df within circle of center @c0 and @radius """
    info(inspect.stack()[0][3] + '()')
    coords = df[['x', 'y']].values
    
    kdtree = cKDTree(coords)
    inds = kdtree.query_ball_point(c0, radius)
    return df.iloc[inds]
    
##########################################################
def filename_from_coords(x, y, heading, ext='jpg'):
    return '_{}_{}_{}.{}'.format(y, x, heading, ext)

##########################################################
def calculate_correlations(dfclulabels, accessibpath, outdir, kdeparam='scott'):
    """Calculate correlation"""
    info(inspect.stack()[0][3] + '()')

    dfaccessib = pd.read_csv(accessibpath)

    if kdeparam < 0: kdebw = 'scott'
    else: kdebw = kdeparam
    
    corrs = {}
    for l in np.unique(dfclulabels.label):
        filtered = dfclulabels[dfclulabels.label == l]
        
        graffloc = np.vstack([filtered.x, filtered.y])
        kernel = stats.gaussian_kde(graffloc, bw_method=kdebw)
        k = ker = kernel(np.vstack([dfaccessib.x.values, dfaccessib.y.values]))

        corrs[l] = []
        for col in sorted(dfaccessib.columns):
            if not 'accessib' in col: continue
            acc = dfaccessib[col].values
            corr, pvalue = scipy.stats.pearsonr(k, acc)
            corrs[l].append(corr)
            info('accessib{}, label{}, corr:{}'.format(col, l, corr))

    accessibs = []
    for col in sorted(dfaccessib.columns):
        if 'accessib' in col: accessibs.append(col)

    corsdf = pd.DataFrame.from_dict(corrs, orient='index', columns=accessibs)
    corsdf.to_csv(pjoin(outdir, 'corrs.csv'))

##########################################################
def correlate_count_and_accessib(dfclulabels, accessibpath, vcoords,
        outdir, kdebw='scott', filterinds=[]):
    """Plot count of graffiti vs accessibility for each vertex"""
    info(inspect.stack()[0][3] + '()')

    accs = pd.read_csv(accessibpath, header=None).values.reshape(-1)
    
    graffloc = np.vstack([dfclulabels.x, dfclulabels.y])
    ker = stats.gaussian_kde(graffloc, bw_method=kdebw)

    info('KERNEL dim:{}, n:{}, neff:{}, factor:{}, cov:{}'.\
            format(ker.d, ker.n, ker.neff, ker.factor, ker.covariance))
    
    if len(filterinds) > 0:
        vcoords = vcoords[filterinds]
        accs = accs[filterinds]
        
    probs = ker(vcoords.T)

    # m = np.max(accs) / 3
    # accs[accs > 2*m] = -3
    # accs[accs > 1*m] = -2
    # accs[accs > 0*m] = -1
    # accs = -accs

    inds = np.argwhere(~np.isnan(accs)).flatten()
    figscale = 6
    fig, axs = plt.subplots(1, 1,
                figsize=(1.2*figscale, figscale))
    plasma = matplotlib.cm.get_cmap('plasma', 100)
    axs.scatter(probs[inds], accs[inds], s=4, alpha=0.2, c=[plasma(.2)])

    corr, _ = scipy.stats.pearsonr(probs[inds], accs[inds])
    axs.tick_params(axis='both', which='major', labelsize='x-large')
    axs.set_xlabel('Graffiti count', fontsize='xx-large')
    axs.set_ylabel('Accessibility', fontsize='xx-large')
    
    accsuff = os.path.splitext(os.path.basename(accessibpath))[0]
    axs.set_title('Accessib {}, KDE {:.02f}, Pearson {:.02f}'. \
            format(accsuff, ker.factor, corr))
    plt.tight_layout(1)
    plt.savefig(pjoin(outdir, '{}_{:.02f}.png'.format(accsuff, ker.factor)))
    return corr

##########################################################
def plot_densities(df, xx, yy, mapx, mapy, outdir, kerbw='scott', filterinds=[]):
    """Plot the densities in the grid @xx, @yy """
    info(inspect.stack()[0][3] + '()')

    labels = np.unique(df.label)

    ncols = len(labels) + 1
    figscale = 4
    fig, axs = plt.subplots(ncols, figsize=(figscale, ncols*figscale))

    # if len(filterinds) > 0:
        # xx = xx[filterinds]
        # yy = yy[filterinds]

    pdfvals = np.ndarray((len(labels), xx.shape[0], xx.shape[1]))
    for j, l in enumerate(labels): # compute the pdf
        filtered = df[(df.label == int(l))]
        pdfvals[j, :, :], _ = compute_pdf_over_grid(filtered.x,
                filtered.y, xx, yy, kerbw)

    im = axs[ 0].scatter(xx, yy, c='gray') # background
    meanpdf = np.mean(pdfvals, axis=0) # mean pdf
    axs[0].plot(mapx, mapy, c='dimgray') # plot border
    # im = axs[ 0].scatter(xx, yy, c=meanpdf) # meand pdf plot
    im = axs[ 0].scatter(xx[filterinds], yy[filterinds], c=meanpdf[filterinds]) # meand pdf plot
    cbar = axs[0].figure.colorbar(im, ax=axs[0], fraction=0.04, pad=0.00)
    axs[0].axis("off")

    minval = 100; maxval = 0
    for j, _ in enumerate(labels):
        aux = pdfvals[j] - meanpdf
        if np.min(aux) < minval: minval = np.min(aux)
        if np.max(aux) > maxval: maxval = np.max(aux)

    if minval*maxval < 0:
        maxval = np.max(np.abs([minval, maxval]))
        minval = -maxval

    cmap = plt.cm.get_cmap('bwr').reversed()
    kld = {}
    for j, l in enumerate(labels):
        jj = j + 1
        vals = pdfvals[j] - meanpdf
        kld[j] = kl_divergence(pdfvals[j] / np.sum(pdfvals[j]),
                meanpdf / np.sum(meanpdf)) # normalized kld

        axs[jj].scatter(xx, yy, c='gray') # background
        axs[jj].plot(mapx, mapy, c='darkgray') # plot border
        im = axs[jj].scatter(xx[filterinds], yy[filterinds], c=vals[filterinds], cmap=cmap,
                vmin=minval, vmax=maxval)
        cbar = axs[jj].figure.colorbar(im, ax=axs[jj], fraction=0.04, pad=0.00)
        axs[jj].axis("off")

    klddf = pd.DataFrame.from_dict(kld, orient='index', columns=['kld'])
    klddf.to_csv(pjoin(outdir, 'klds.csv'), index_label='type')
    plt.tight_layout(2)
    labels = ['mean', 'typeA', 'typeB', 'typeC']
    pads = [.1, .1, .6, .1]

    if type(kerbw) == str: pref = 'ker_{}_'.format(kerbw)
    else: pref = 'ker_{:.02f}_'.format(kerbw)

    export_individual_axis(axs, fig, labels, outdir, pad=pads,
            prefix=pref, fmt='png')

##########################################################
def get_knn_ratios(labelsdf, vcoords, k, outdir):
    """Get ratios based on the kNN"""
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'ratios.csv')
    if os.path.exists(outpath):
        x =  pd.read_csv(outpath, header=None).values
        return x[:, :3], x[:, -1]

    labels = np.unique(labelsdf.label)
    nvertices = vcoords.shape[0]
    graffcoords = labelsdf[['x', 'y']].values
    tree = cKDTree(graffcoords)

    # c0 = [-46.6333, -23.5505] # Sao paulo
    # distsall, indsall = tree.query([c0], k=3)

    distsall, indsall = tree.query(vcoords, k=k)
    radius = np.zeros(nvertices, dtype=float)
    ratios = np.zeros((nvertices, len(labels)), dtype=float)

    for i, inds in enumerate(indsall):
        radius[i] = distsall[i][-1]
        ser = labelsdf.iloc[inds]
        counts = np.unique(ser.label, return_counts=True)
        
        for _ind, _count in zip(counts[0], counts[1]):
            arrayid = _ind - 1 # 0-based index
            ratios[i, arrayid] = _count
        # if i > 1000: break

    ratios /= k
    df = pd.DataFrame(ratios)
    df['radius'] = radius
    df.to_csv(outpath, header=False, index=False)
    return ratios, radius #nverticesx3, nvertices

##########################################################
# @jit
def multivariate_normal(x, mean, cov):
    """P.d.f. of the multivariate normal when the covariance matrix is pos.def.
    Source: wikipedia"""
    ndims = len(mean)
    B = x - mean
    return (1. / (np.sqrt((2 * np.pi)**ndims * np.linalg.det(cov))) *
            np.exp(-0.5*(np.linalg.solve(cov, B).T.dot(B))))

##########################################################
def test_multivariate_normal(x):
    n = 3 # sigma. 3 sigma cover 98%
    m = 100
    d = (2*n / m) * (2*n / m) #area of each tile
    mean = np.array([0, 0])
    cov = np.eye(2)


    xx, yy = np.mgrid[-n:+n:(m*1j), -n:+n:(m*1j)]
    acc = 0
    for rowx, rowy in zip(xx, yy): #integration by hand
        for elx, ely in zip(rowx, rowy):
            acc += multivariate_normal(np.array([elx, ely], mean, cov))
    acc *= d # V=h*b
    return acc

##########################################################
def find_tile_idx(pt, xx, yy, dx, dy):
    """Find tile index """
    gridmin = np.array([xx[0, 0], yy[0, 0]])
    delta = pt - gridmin
    return int(delta[0] // dx), int(delta[1] // dy)

##########################################################
def gaussian_smooth(coords, refcoords, ratios, radius, nsigma, outpath):
    """Gaussian smooth for every type"""
    info(inspect.stack()[0][3] + '()')

    diffr = .001 # diffusion radius for density estimation
    if os.path.exists(outpath):
        return pickle.load(open(outpath, 'rb'))

    tree = cKDTree(refcoords)
    funcs = np.ndarray(len(refcoords), dtype=object)

    g = np.zeros(len(coords), dtype=float)

    def myfun(x, mean, radius, nsigma, ratio, r):
        R = radius
        sigma = radius / nsigma
        cov = np.eye(2) * (sigma**2)
        peak =  multivariate_normal(x, mean, cov)

        # return (peak * ratio) # TODO: remove it

        if R <= r: return (peak * ratio) # very dense region

        inds = np.array(tree.query_ball_point([x], R)[0])

        if len(inds) == 0: return 0 # very sparse region

        pts = GeoSeries([ Point(p[0], p[1]) for p in refcoords[inds] ])
        circles = pts.buffer(r)
        occupied = circles.unary_union.area
        total = np.pi * R**2
        factor = min(occupied / total, 1)
        return (peak * ratio) * factor

    for i in range(len(refcoords)): # parameterized gaussians functions
        funcs[i] = partial(myfun, mean=refcoords[i, :], radius=radius[i],
                nsigma=nsigma, ratio=ratios[i], r=diffr)

    for i in range(len(coords)): # filtering by...
        info('i:{}'.format(i))
        c0 = np.array([coords[i, :]]) # ...distance
        g[i] = funcs[i](c0[0])

    pickle.dump(g, open(outpath, 'wb'))
    return g

##########################################################
def run_experiment_from_list(params):
    return gaussian_smooth(*params)

##########################################################
def gaussian_smooth_all(coords, vcoords, ratiosall, radius, nsigma,
        outdir, suff='', nprocs=1):
    nratios = ratiosall.shape[1]
    params = []
    for i in range(ratiosall.shape[1]):
        outpath = pjoin(outdir, 'gaussian_{}{}.pkl'.format(suff, i))
        params.append([coords, vcoords, ratiosall[:, i], radius, nsigma, outpath])

    return Pool(nprocs).map(run_experiment_from_list, params)

    # gs = []
    # for param in params:
        # gs.append(run_experiment_from_list(param))

##########################################################
def plot_gaussians(gins, n, outdir):
    for i, gin in enumerate(gins):
        fig, ax = plt.subplots(figsize=(10, 10))
        outpath = pjoin(outdir, 'gaussian_{}.png'.format(i))
        g = gin.reshape(n, n)
        im = ax.imshow(g.T, origin='lower')
        fig.colorbar(im)
        plt.savefig(outpath)

        fig, ax = plt.subplots(figsize=(10, 10))
        outpath = pjoin(outdir, 'gaussian_log{}.png'.format(i))
        im = ax.imshow(np.log(g.T+.0001), origin='lower')
        fig.colorbar(im)
        plt.savefig(outpath)

##########################################################
def get_vertices_above_density(query, refcoords, r):
    """Get indices of graph vertices which have at least one graff
    occurrence nearby"""
    info(inspect.stack()[0][3] + '()')

    from sklearn.neighbors import BallTree
    bt = BallTree(np.deg2rad(refcoords), metric='haversine')

    counts = bt.query_radius(
            np.deg2rad(np.c_[query[:, 0], query[:, 1]]), r=r/R,
            count_only=True)
    _, x = np.unique(counts, return_counts=True)
    
    return np.where(counts > 0)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nneighbours', default=3, type=int, help='Num neighbours')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    annotdir = './data/20200202-types/20200209-8003_annot/'
    graphmlpath = './data/20200202-types/sp.graphml'
    clupath = './data/20200202-types/20200222-infomap.clu'
    shppath = './data/20200202-types/20200224-shp/'
    cluareaspath = './data/20200202-types/20200222-infomap_areas.csv'
    outlabels = pjoin(args.outdir, 'labels.csv')
    labelsclu = 'data/labels_and_clu_nodupls.csv'
    vcoordspath = 'data/vcoords.csv'
    c0 = [-46.6333, -23.5505] # Sao paulo

    shppath = './data/20200224-shp/'
    accpath = './data/20200630-accessib/acc15.txt'

    labelsdf = compile_labels(annotdir, labelsclu) # Do this for each annotator
    labelsdf, vcoords = parse_infomap_results(graphmlpath, clupath, labelsdf,
            'er', labelsclu, vcoordspath)
    ngraff = len(labelsdf)
    
    grcoords = [[labelsdf.iloc[i].x, labelsdf.iloc[i].y] for i in range(ngraff)]
    grcoords = np.array(grcoords)

    # plot_occurrences_density(vcoords, labelsdf, args.outdir)
    minr = 500
    indsdens = [] #indsdens = get_vertices_above_density(vcoords, grcoords, minr)
    
    # for kerbw in np.arange(.05, .41, .05):
        # info('kerbw:{}'.format(kerbw))
        # corr = correlate_count_and_accessib(labelsdf, accpath, vcoords, args.outdir, kerbw, indsdens)
        # info('corr:{}'.format(corr))
    # return

    # plot_types(clupath, shppath, labelsclu, args.outdir)
    # clus = sorted(np.unique(labelsdf.cluster))
    # lbls = sorted(np.unique(labelsdf.label))
    # results = get_ratios_by_community(labelsdf, clus, normalized=True)
    # plot_stacked_bar_types(results, len(clus), len(lbls),
            # palettehex3, args.outdir)
    # plot_counts_normalized(labelsclu, cluareaspath, args.outdir)
    # plot_venn(labelsclu, args.outdir)

    # Kernel density estimation
    info('Elapsed time:{}'.format(time.time()-t0))
    ntilesx = ntilesy = 100
    xx, yy, dx, dy = create_meshgrid(labelsdf.x, labelsdf.y,
            nx=ntilesx, ny=ntilesy, relmargin=.1)

    nsigma = 3
    ratios, radius = get_knn_ratios(labelsdf, vcoords, args.nneighbours, args.outdir)
    gs = gaussian_smooth_all(vcoords, vcoords, ratios, radius, nsigma,
            args.outdir, suff='vcoords_', nprocs=3) # for vertex coords
    return

    # coords = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    # gs = gaussian_smooth_all(coords, vcoords, ratios, radius, nsigma,
            # args.outdir, suff='tiles_', nprocs=3)
    # plot_gaussians(gs, ntilesx, args.outdir)

    kerbw = .3

    pdf, _ = compute_pdf_over_grid(labelsdf.x, labelsdf.y, xx, yy, kerbw)
    flattened = np.array([[x_, y_] for x_, y_ in zip(xx.flatten(), yy.flatten())])
    indsdensflat = get_vertices_above_density(flattened, grcoords, minr)
    ii = (indsdensflat[0]/ntilesy).astype(int)
    jj = (indsdensflat[0]%ntilesy).astype(int)
    # indsdens = (ii, jj)

    # plot_contours(pdf, labelsdf.x, labelsdf.y, xx, yy, args.outdir)
    # plot_surface(pdf, labelsdf.x, labelsdf.y, xx, yy, args.outdir)
    # plot_wireframe(pdf, labelsdf.x, labelsdf.y, xx, yy, args.outdir)
    # plot_hist2d(labelsdf.x, labelsdf.y, args.outdir)
    # mapx, mapy = geo.get_shp_points(shppath)
    # plot_densities(labelsdf, xx, yy, mapx, mapy, args.outdir, kerbw, indsdens)

    # for kerbw in np.arange(.05, .41, .05):
    for kerbw in [.3]:
        info('kerbw:{}'.format(kerbw))
        corr = correlate_count_and_accessib(labelsdf, accpath, vcoords, args.outdir, kerbw, indsdens)
        info('corr:{}'.format(corr))

    # calculate_correlation(df, accessibpath, args.outdir, .3)

##########################################################
if __name__ == "__main__":
    main()
