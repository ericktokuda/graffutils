"""Analyze results from deeplab and get the best checkpoint
"""

import os
import numpy as np
from logging import debug, info
from os.path import join as pjoin
import cv2
import matplotlib.pyplot as plt

class DeeplabAnalyzer:
    def __init__(self):
        np.random.seed(0)

    def run(self, args, outdir):
        if len(args) < 1:
            info('Please provide the stdout from deeplab results. Aborting...')
            return
        elif not os.path.exists(args[0]):
            info('Please check if {} exist'.format(args))
            return

        fh = open(args[0], 'r')

        res = []
        while True:
           aux = fh.readline()
           ckpt = int(aux.replace('model.ckpt-', '').replace('.meta', ''))
           print(ckpt)
           aux = fh.readline()
           idx = aux.find('class_0')
           aux = aux[idx+12:]
           idx = aux.find(']')
           aux = aux[:idx]
           iou0 = float(aux)

           aux = fh.readline()
           idx = aux.find('class_1')
           aux = aux[idx+12:]
           idx = aux.find(']')
           aux = aux[:idx]
           iou1 = float(aux)

           res.append([ckpt, iou0, iou1, (iou0+iou1)/2])


           if ckpt == 9730: break

        fh.close()


        fh = open(pjoin(outdir, 'iou.csv', 'w'))
        for r in res:
           fh.write(','.join([ str(x) for x in r ]) + '\n')
        fh.close()

        res = np.array(res)
        plt.scatter(res[:, 0], res[:, 3])
        plt.scatter(res[:, 0], res[:, 2])
        plt.grid()
        plt.show()
