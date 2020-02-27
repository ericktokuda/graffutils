"""Parse .clu results from infomap

"""

import os
import numpy as np
from logging import debug, info
from os.path import join as pjoin
import cv2
import pandas as pd
import igraph
from scipy.spatial import cKDTree

class InfomapParser:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 2:
            info('Please provide (1)cluster, (2)graphml and (3)labels paths. Aborting...')
            return

        for a in args:
            if not os.path.exists(a):
                info('Please check if {} exists'.format(a))
                return

        clupath = args[0]
        graphmlpath = args[1]
        labelspath = args[2]

        g = igraph.Graph.Read(graphmlpath)
        ########################################################## load cluster
        clusterdf = pd.read_csv(clupath, sep=' ', skiprows=[0, 1],
                         names=['id', 'cluster','flow'])

        clusterdf = clusterdf.sort_values(by=['id'], inplace=False)

        ########################################################## load graffiti
        objsdf = pd.read_csv(labelspath, index_col='id')

        pd.set_option("display.precision", 8)

        coords_objs = np.zeros((len(objsdf), 2))
        for idx, row in objsdf.iterrows():
            coords_objs[idx, 0] = row.x
            coords_objs[idx, 1] = row.y

        coords_nodes = np.array([[float(x), float(y)] for x, y in zip(g.vs['x'], g.vs['y'])])
        kdtree = cKDTree(coords_nodes)
        dists, inds = kdtree.query(coords_objs)
        allclusters = np.array(clusterdf.cluster.tolist())
        clusters = allclusters[inds]

        objsdf['cluster'] = clusters
        objsdf.to_csv(pjoin(outdir, 'clusters.csv'))

        for cl in np.unique(clusters):
            objsdf[objsdf.cluster == cl].\
                to_csv(pjoin(outdir, 'region{}_ids.txt'.format(cl)), columns=[],
                       index=True, header=False)
