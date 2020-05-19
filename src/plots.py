#!/usr/bin/env python3
"""Bar plots of the graffiti occurrences
"""

import argparse
import os
import numpy as np
from os.path import join as pjoin
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import igraph
from matplotlib.ticker import FormatStrFormatter

from src.utils import info

class LabelPlotter:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 1:
            info('Please provide the (1)csv file with a field cluster and'\
                 ' a field label. Aborting...')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args))
            return

        labelspath = args[0]
        cluareaspath = args[1]
        df = pd.read_csv(labelspath, index_col='id')
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

        coloursrev = []
        for i in range(len(palette)):
            coloursrev.append(colours[len(palette) - 1 - i, :])

        # Plot by type
        fig, ax = plt.subplots(1, nlabels,
                               figsize=(nlabels*plotsize, 1*plotsize),
                               squeeze=False)

        clustersums = np.zeros((nclusters, nlabels))
        for i, cl in enumerate(clusters):
            aux = df[df.cluster == cl]
            for j, l in enumerate(labels):
                clustersums[i, j] = len(aux[aux.label == l])

        labelsmap = {1: 'A', 2: 'B', 3: 'C'}
        for i, l in enumerate(labels):
            data = df[df.label == l]
            ys = np.zeros(nclusters)
            for k, cl in enumerate(clusters):
                ys[k] = len(data[data.cluster == cl]) / np.sum(clustersums[k, :])

            barplot = ax[0, i].barh(list(reversed(clusters_str)), list(reversed(ys)),
                                    color=coloursrev)

            ax[0, i].axvline(x=len(df[df.label == l])/totalrows, linestyle='--')
            ax[0, i].text(len(df[df.label == l])/totalrows + 0.05,
                          -0.7, 'ref', ha='center', va='bottom',
                          rotation=0, color='royalblue',
                          fontsize='large')
            reftick = len(df[df.label == l])/totalrows
            ax[0, i].set_xlim(0, 1)
            ax[0, i].set_title('Ratio of Type {} within communities'.\
                                format(r"$\bf{" + str(labelsmap[l]) + "}$"),
                               size='large', pad=30)
            ax[0, i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[0, i].set_xticks([0, .25, 0.5, .75, 1.0])
            ax[0, i].spines['top'].set_color('gray')
            ax[0, i].xaxis.set_ticks_position('top')
            ax[0, i].tick_params(axis='x', which='both', length=0, colors='gray',
                                 labelsize='large')
            ax[0, i].tick_params(axis='y', which='both', labelsize='large')
            ax[0, i].xaxis.grid(True, alpha=0.4)
            ax[0, i].set_axisbelow(True)

            def autolabel(rects, ys):
                for idx, rect in enumerate(rects):
                    height = rect.get_height()
                    # print(rect.get_x(), height)
                    ax[0, i].text(rect.get_width()-0.05,
                                  rect.get_y() + rect.get_height()/2.-0.18,
                                  '{:.2f}'.format(ys[idx]), color='white',
                                  ha='center', va='bottom', rotation=0,
                                  fontsize='large')

            autolabel(barplot, list(reversed(ys)))
            # ax[0, i].axis("off")
            for spine in ax[0, i].spines.values():
                spine.set_edgecolor('dimgray')
            ax[0, i].spines['bottom'].set_visible(False)
            ax[0, i].spines['right'].set_visible(False)
            ax[0, i].spines['left'].set_visible(False)

        # plt.box(False)
        # fig.suptitle('Ratio of graffiti types inside each cluster', size='x-large')
        plt.tight_layout(pad=5)
        plt.savefig(pjoin(outdir, 'count_per_type.pdf'))
        fig.clear()

        # Plot overall count
        fig, ax = plt.subplots(1, 1, figsize=(plotsize, plotsize),
                               squeeze=False)

        counts = np.zeros(nclusters, dtype=int)
        countsnorm = np.zeros(nclusters)
        areas = pd.read_csv(cluareaspath)

        for i, cl in enumerate(clusters):
            data = df[df.cluster == cl]
            counts[i] = len(data)
            points = data[['x', 'y']].values

            countsnorm[i] = counts[i] / areas.iloc[i]

        cumsum = 0.0
        axs = []
        for i, cl in enumerate(clusters):
            axs.append(ax[0, 0].barh(0, counts[i], 0.5, left=cumsum,
                                     label=clusters_str[i],
                                     color=colours[i]))
            cumsum += counts[i]
        # search all of the bar segments and annotate
        for j in range(len(axs)):
            for i, patch in enumerate(axs[j].get_children()):
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = patch.get_height() + bl[1]
                ax[0, 0].text(x,y, '{}'.format(counts[j]), ha='center')

        ax[0, 0].set_ylim(0, 3)
        ax[0, 0].legend()
        fig.patch.set_visible(False)
        ax[0, 0].axis('off')
        plt.savefig(pjoin(outdir, 'counts.pdf'))

        fig, ax = plt.subplots(1, 1, figsize=(2*plotsize, plotsize),
                               squeeze=False)
        yfactor = 1
        ax[0, 0].bar(clusters_str, countsnorm / yfactor, color=colours)
        ax[0, 0].set_ylabel('Normalized count of graffitis')
        ax[0, 0].set_xlabel('Community')
        for spine in ax[0, i].spines.values():
            spine.set_edgecolor('dimgray')
        ax[0, i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax[0, i].spines['top'].set_visible(False)
        ax[0, i].spines['right'].set_visible(False)
        ax[0, i].yaxis.grid(True, alpha=0.4)
        ax[0, i].set_axisbelow(True)
        # ax[0, i].spines['left'].set_visible(False)

        plt.savefig(pjoin(outdir, 'countsnormalized.pdf'))

class MapGenerator:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 4:
            info('Please provide the (1)graphml, (2)shapefile, \
                 (3)labels csv, (4) clu file. '\
                 'Aborting...')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args))
            return

        graphmlpath = args[0]
        shppath = args[1]
        labelspath = args[2]
        clupath = args[3]
        df = pd.read_csv(labelspath, index_col='id')
        totalrows = len(df)

        g = igraph.Graph.Read(graphmlpath)
        g.simplify()
        g.to_undirected()
        coords = [(float(x), -float(y)) for x, y in zip(g.vs['x'], g.vs['y'])]

        fh = open(clupath)
        lines = fh.read().strip().split('\n')

        aux = np.zeros(g.vcount())
        for l in lines[2:]:
            arr = l.split(' ')
            aux[int(arr[0])-1] = int(arr[1])
        g.vs['clustid'] = aux
        fh.close()
        info('Clusters found in infomap: {}'.format(np.unique(g.vs['clustid'])))

        for attr in ['ref', 'highway', 'osmid', 'id']:
            del(g.vs[attr])
        for attr in g.es.attributes():
            del(g.es[attr])

        fig, ax = plt.subplots(1, 2, figsize=(15, 12), squeeze=False) # Plot contour

        gdf = gpd.read_file(shppath)
        shapefile = gdf.geometry.values[0]
        xs, ys = shapefile.exterior.xy
        ax[0, 0].plot(xs, ys, c='dimgray')

        labelsdf = pd.read_csv(labelspath)
        clusters = np.unique(labelsdf.cluster)
        clusters_str = ['C{}'.format(cl) for cl in clusters]
        nclusters = len(clusters)
        labels = np.unique(labelsdf.label)
        nlabels = len(labels)

        markers = ['$A$', '$B$', '$C$']
        ss = [30, 35, 35]
        # edgecolours = ['#914232', '#917232', '#519132']
        edgecolours = ['#0000FF', '#FF0000', '#00FF00']
        visual = [ dict(marker=m, s=s, edgecolors=e) for m,s,e in \
                  zip(['o', 'o', 'o'], ss, edgecolours)]
                  # zip(markers, ss, edgecolours)]

        for i, l in enumerate(labels):
            data = labelsdf[labelsdf.label == l]
            ax[0, 0].scatter(data.x, data.y, c=edgecolours[i],
                             label='Type ' + markers[i],
                             alpha=0.6,
                             # linewidths=0.2,
                             # edgecolor=(0.3, 0.3, 0.3, 1),
                             **(visual[i]))
        
        fig.patch.set_visible(False)
        ax[0, 0].axis('off')
        # -46.826198999999995 -46.36508400000003 -24.008430999701822 -23.356292999687376

        ax[0, 0].legend(loc=(0.6, 0.12), title='Graffiti types',
                        fontsize='xx-large', title_fontsize='xx-large')

        ##########################################################
        palette = np.array([
            [27.0,158,119],
            [217,95,2],
            [117,112,179],
            [231,41,138],
            [102,166,30],
            [230,171,2],
        ])
        alpha = 0.6
        palette /= 255.0
        colours = np.zeros((palette.shape[0], 4), dtype=float)
        colours[:, :3] = palette
        colours[:, -1] = alpha
        palette = colours

        coords = [(float(x), -float(y)) for x, y in zip(g.vs['x'], g.vs['y'])]
        coordsnp = np.array([[float(x), float(y)] for x, y in zip(g.vs['x'], g.vs['y'])])
        # vcolours = [palette[int(v['clustid']-1)] for v in g.vs() ]

        clustids = np.array(g.vs['clustid']).astype(int)
        ecolours = np.zeros((g.ecount(), 4), dtype=float)
        lines = np.zeros((g.ecount(), 2, 2), dtype=float)

        for i, e in enumerate(g.es()):
            srcid = int(e.source)
            tgtid = int(e.target)

            ecolours[i, :] = palette[int(g.vs[srcid]['clustid'])-1]

            if g.vs[tgtid]['clustid'] != g.vs[tgtid]['clustid']:
                ecolours[i, 3] = 0.0

            lines[i, 0, 0] = g.vs[srcid]['x']
            lines[i, 0, 1] = g.vs[srcid]['y']
            lines[i, 1, 0] = g.vs[tgtid]['x']
            lines[i, 1, 1] = g.vs[tgtid]['y']

        lc = mc.LineCollection(lines, colors=ecolours, linewidths=0.5)
        ax[0, 1].add_collection(lc)
        ax[0, 1].autoscale()

        gdf = gpd.read_file(shppath)
        shapefile = gdf.geometry.values[0]
        xs, ys = shapefile.exterior.xy
        ax[0, 1].plot(xs, ys, c='dimgray')
        
        fig.patch.set_visible(False)
        ax[0, 1].axis('off')
        plt.tight_layout(pad=10)

        handles = []
        for i, p in enumerate(palette):
            handles.append(mpatches.Patch(color=palette[i, :], label='C'+str(i+1)))

        # plt.legend(handles=handles)
        ax[0, 1].legend(handles=handles, loc=(.6, .12), title='Communities',
                        fontsize='xx-large', title_fontsize='xx-large')

        extent = ax[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(pjoin(outdir, 'map_types.png'), bbox_inches=extent.expanded(1.0, 1.0))

        extent = ax[0, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(pjoin(outdir, 'map_comm.png'), bbox_inches=extent.expanded(1.0, 1.0))

        plt.savefig(pjoin(outdir, 'maps.png'))

class MasksGenerator:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 2:
            info('Please provide (1)wkt and (2)images dir. Aborting...')
            return
        elif not os.path.exists(args[0]) or not os.path.exists(args[1]):
            info('Please check if {} exist'.format(args))
            return

        wktdir = args[0]
        imgsdir = args[1]
        samplesz = 100
        minarea = 4900

        acc = 0

        files = sorted(os.listdir(wktdir))
        np.random.shuffle(files)

        fh = open('/tmp/analyzedimgs.txt', 'w')
        for i, f in enumerate(files):
            if acc == samplesz: break
            if not f.endswith('.wkt'): continue
            wktpath = pjoin(wktdir, f)
            if os.stat(wktpath).st_size == 0: continue
            info('f:{}'.format(f))

            maskpath = pjoin(outdir, f.replace('.wkt', '_mask.jpg'))
            edgespath = pjoin(outdir, f.replace('.wkt', '_edges.jpg'))
            polypath = pjoin(outdir, f.replace('.wkt', '_poly.jpg'))
            img = cv2.imread(pjoin(imgsdir, f.replace('.wkt', '.jpg')))

            polys = self.parse_wkt(wktpath)
            # print(wktpath, polys)
            polys = self.filter_polys_by_area(polys, minarea)
            if len(polys) == 0: continue
            # draw_mask_from_wkt(polys, maskpath, img)
            # draw_edge_mask_from_wkt(polys, edgespath, img)
            # print(img)
            self.draw_polygon_from_wkt(polys, polypath, img)
            fh.write(f.replace('.wkt', '') + '\n')
            acc += 1
        fh.close()

    ##########################################################
    def draw_edge_mask_from_wkt(self, polys, maskpath, imgorig=[]):
        backgrcolor = 0
        foregrcolor = [256, 256, 256, 0.5]

        if len(imgorig) == 0:
            img = np.ones((640, 640), dtype=np.uint8) * backgrcolor

            img = cv2.Canny(imgorig, 100, 200)
            stencil = np.zeros(img.shape).astype(img.dtype)
            cv2.fillPoly(stencil, polys, foregrcolor, lineType=8, shift=0)
            img = cv2.bitwise_and(img, stencil)
            cv2.imwrite(maskpath, img)

    ##########################################################
    def parse_wkt(self, wktpath):
        wktfh = open(wktpath)
        f = shapely.wkt.loads(wktfh.read())

        polys = []
        for poly in f:
            coords = []
            for x, y in zip(*poly.exterior.coords.xy):
                coords.append([int(x), int(y)])
            polys.append(np.array(coords))

        wktfh.close()
        return np.array(polys)

    def draw_mask_from_wkt(self, polys, maskpath, imgorig=[]):
        backgrcolor = 0
        foregrcolor = [256, 256, 256, 0.5]

        if len(imgorig) == 0:
            img = np.ones((640, 640), dtype=np.uint8) * backgrcolor

            img = imgorig.copy()
            stencil = np.zeros(img.shape).astype(img.dtype)
            cv2.fillPoly(stencil, polys, foregrcolor, lineType=8, shift=0)
            img = cv2.bitwise_and(img, stencil)
            cv2.imwrite(maskpath, img)

    ##########################################################
    def draw_polygon_from_wkt(self, polys, maskpath, imgorig=[]):
        edgecolor = (0, 0, 255)
        if len(imgorig) == 0:
            img = np.ones((640, 640), dtype=np.uint8) * backgrcolor

        img = imgorig.copy()

        for p in polys:
            cv2.polylines(img, np.int32([p]), 1, edgecolor, thickness=3)

        cv2.imwrite(maskpath, img)

    def filter_polys_by_area(self, polys, minarea):
        todel = []
        for i, poly in enumerate(polys):
            x = Polygon(poly)
            polyarea = Polygon(poly).area
            if polyarea < minarea:
                todel.append(i)

        x = np.delete(polys, todel, axis=0)
        return x

    ##########################################################

##########################################################
def generate_rgb_colors(n):
    hsvcols = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgbcols = map(lambda x: colorsys.hsv_to_rgb(*x), hsvcols)
    return list(rgbcols)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphml', required=True, help='Graphml path')
    args = parser.parse_args()

    g = igraph.Graph.Read_GraphML(args.graphml)
    g.to_undirected()
    info('Loaded')
    if 'x' in g.vs.attributes():
        g.vs['x'] = np.array(g.vs['x']).astype(float)
        g.vs['y'] = -np.array(g.vs['y']).astype(float)

    info('g.vcount():{}'.format(g.vcount()))
    info('g.ecount():{}'.format(g.ecount()))
    del g.vs['id'] # Avoid future warnings
    attrs = g.es.attributes()
    for attr in attrs: del g.es[attr]
    igraph.write(g, '/tmp/foo.graphml', 'graphml')
    info('saved')
    info('Reload')
    g = igraph.read('/tmp/foo.graphml')
    g.simplify()
    # dendr = g.community_fastgreedy()
    # clusters = dendr.as_clustering()

    # clusters = g.community_leading_eigenvector(clusters=5)
    # clusters = g.community_edge_betweenness(clusters=5) # get stucked
    dendr = g.community_walktrap()
    clusters = dendr.as_clustering()
    # get the membership vector
    membership = clusters.membership
    colorlist = generate_rgb_colors(len(np.unique(membership)))

    igraph.plot(g, target='/tmp/communities.pdf', vertex_size=4,
                bbox = (2000, 2000),
                vertex_color=[ colorlist[m] for m in membership ],
                vertex_frame_width=0)


if __name__ == "__main__":
    main()

