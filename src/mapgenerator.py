#!/usr/bin/env python3
"""Generate map based on infomap results

Run this after running:
./Infomap ~/temp/citysp.net /tmp/ --directed --clu --tree --bftree --map
"""

import os
import numpy as np
from logging import debug, info
from os.path import join as pjoin
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.patches as mpatches
import geopandas as gpd
import shapely
import igraph

class MapGenerator:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 3:
            info('Please provide the (1)graphml, (2)labels csv, (3) clu file. '\
                 'Aborting...')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args))
            return

        graphmlpath = args[0]
        labelspath = args[1]
        clupath = args[2]
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

        fig, ax = plt.subplots(figsize=(10, 15)) # Plot contour
        gdf = gpd.read_file('/home/frodo/temp/citysp_shp/')
        shapefile = gdf.geometry.values[0]
        xs, ys = shapefile.exterior.xy
        ax.plot(xs, ys, c='dimgray')

        labelsdf = pd.read_csv(labelspath)
        clusters = np.unique(labelsdf.cluster)
        clusters_str = ['C{}'.format(cl) for cl in clusters]
        nclusters = len(clusters)
        labels = np.unique(labelsdf.label)
        nlabels = len(labels)

        markers = ['$A$', '$B$', '$C$']
        ss = [25, 30, 30]
        edgecolours = ['#914232', '#917232', '#519132']
        visual = [ dict(marker=m, s=s, edgecolors=e) for m,s,e in \
                  zip(markers, ss, edgecolours)]

        for i, l in enumerate(labels):
            data = labelsdf[labelsdf.label == l]
            ax.scatter(data.x, data.y, c='gray', linewidths=[1,1,1],
                       label='Type ' + markers[i],
                       **(visual[i]))
        
        fig.patch.set_visible(False)
        ax.axis('off')

        # plt.tight_layout()
        # plt.legend(loc='lower right', title='Graffiti types')
        # plt.savefig(pjoin('/tmp/types.png'))

        ##########################################################
        palette = np.array([
            [27.0,158,119],
            [217,95,2],
            [117,112,179],
            [231,41,138],
            [102,166,30],
            [230,171,2],
        ])
        alpha = 0.7
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

        fig, ax = plt.subplots(figsize=(10, 15))
        lc = mc.LineCollection(lines, colors=ecolours, linewidths=0.5)
        ax.add_collection(lc)
        ax.autoscale()

        gdf = gpd.read_file('/home/frodo/temp/citysp_shp/')
        shapefile = gdf.geometry.values[0]
        xs, ys = shapefile.exterior.xy
        ax.plot(xs, ys, c='dimgray')
        
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.tight_layout()

        handles = []
        for i, p in enumerate(palette):
            handles.append(mpatches.Patch(color=palette[i, :], label='C'+str(i+1)))

        # plt.legend(handles=handles)

        plt.legend(handles=handles, loc='lower right', title='Communities')
        plt.savefig(pjoin('/tmp/foo.png'))
