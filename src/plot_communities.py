#!/usr/bin/env python3
"""Plot input graphml
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import igraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import colorsys

def generate_rgb_colors(n):
    hsvcols = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgbcols = map(lambda x: colorsys.hsv_to_rgb(*x), hsvcols)
    return list(rgbcols)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphml', required=True, help='Graphml path')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)
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
    dendr = g.community_fastgreedy()
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

