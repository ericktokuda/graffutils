#!/usr/bin/env python3
"""Plot correlations """

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
from myutils import info, create_readme, plot

palettehex3 = ['#e41a1c','#377eb8','#e6e600','#984ea3','#ff7f00','#4daf4a','#a65628','#f781bf','#777777']

##########################################################
def main():
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrs', required=True, help='Correlations in csv fmt')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    df = pd.read_csv(args.corrs)

    # Render latex tables
    kernelbw = 0.3
    accstep = 22

    meth = 'selfavoiding'
    filtered = df.loc[(df.accmethod == meth) & (df.kernelbw == kernelbw) & \
                      (df.accstep == accstep) & (df.community != -1)]
    print(filtered.to_latex(columns=['community', 'type1', 'type2', 'type3'],
                            index=False, float_format='{:0.2f}'.format))

    meth = 'selfavoiding'
    filtered = df.loc[(df.accmethod == meth) & (df.kernelbw == kernelbw) & \
                      (df.accstep == accstep) & (df.community != -1)]
    print(filtered.to_latex(columns=['community', 'type1', 'type2', 'type3'],
                            index=False, float_format='{:0.2f}'.format))

    # Plot line
    figscale = 4
    accmeths = ['selfavoiding', 'randomwalk']
    labels = ['All', 'Type A', 'Type B', 'Type C']
    for i, meth in enumerate(accmeths):
        outpath = pjoin(args.outdir, 'corrXaccstep_ker{}_{}.pdf'.\
                        format(kernelbw, meth))
        fig, ax = plt.subplots(figsize=(1.3*figscale, figscale))
        for grtype in range(1, 4):
                filtered = df.loc[(df.accmethod == meth) & (df.kernelbw == kernelbw) \
                                  & (df.community == 2)]
                col = 'type{}'.format(grtype)
                label = labels[grtype]
                ax.plot(filtered.accstep,
                     filtered[col], label=label, c=palettehex3[grtype+5])
                ax.set_xlabel('Accessibility step')
                ax.set_ylabel('Pearson correlation')
                ax.set_ylim([0, .6])
                # ax.set_title(meth)
                ax.legend()
        plt.tight_layout()
        plt.savefig(outpath)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
