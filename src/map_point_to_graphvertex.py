#!/usr/bin/env python3
"""Map point->graphvertex
"""

import argparse
import logging
import time
from os.path import join as pjoin
from logging import debug, info

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphml', required=True, help='Map in graphml format')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)



    info('Elapsed time:{}'.format(time.time()-t0))
if __name__ == "__main__":
    main()

