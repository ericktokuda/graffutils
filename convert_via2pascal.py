#!/usr/bin/env python3
"""Convert via format annoation to pascal. Using fixed size of 640x640

"""

import os
import argparse
import json
import numpy as np
import pascal_voc_writer


def args_ok(args):
    for arg in vars(args):
        if getattr(args, arg) == None: return False
        return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputjson', help='Json file to be parsed')
    parser.add_argument('--outdir', help='Output directory')
    args = parser.parse_args()

    if not args_ok(args):
        parser.print_help()
        return
    
    fh = open(args.inputjson)
    myjson = json.load(fh)

    for _, entry in myjson.items():
        convert_entry(entry, args.outdir)

    fh.close()

def convert_entry(entry, outdir):
    imgfilename = entry['filename']
    dumbpath = os.path.join('/tmp/', imgfilename)
    regions = entry['regions']

    w = pascal_voc_writer.Writer(dumbpath, 640, 640)
    for _, region in regions.items():
        shape = region['shape_attributes']
        xs = shape['all_points_x']
        ys = shape['all_points_y']
        xmin = np.min(xs)
        ymin = np.min(ys)
        xmax = np.max(xs)
        ymax = np.max(ys)

        w.addObject('scribble', xmin, ymin, xmax, ymax)
    outxml = os.path.join(outdir, os.path.splitext(imgfilename)[0] + '.xml')
    w.save(outxml)

if __name__ == "__main__":
    main()

