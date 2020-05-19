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

