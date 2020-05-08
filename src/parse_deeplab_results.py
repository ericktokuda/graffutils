"""
Script to find the best iou from the deeplab log
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fh = open('del.txt', 'r')

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
    # print(iou0)

    aux = fh.readline()
    idx = aux.find('class_1')
    aux = aux[idx+12:]
    idx = aux.find(']')
    aux = aux[:idx]
    iou1 = float(aux)
    # print(iou1)

    res.append([ckpt, iou0, iou1, (iou0+iou1)/2])

    if ckpt == 9730: break

fh.close()

fh = open('/tmp/iou.csv', 'w')
for r in res:
    fh.write(','.join([ str(x) for x in r ]) + '\n')
fh.close()

res = np.array(res)
plt.scatter(res[:, 0], res[:, 3])
plt.scatter(res[:, 0], res[:, 2])
plt.grid()
plt.show()
