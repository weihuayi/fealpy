#!/usr/bin/env python3
#

import sys 
import math

import numpy as np
import matplotlib.pyplot as plt

# fealpy module
from fealpy.functionspace import FourierSpace
from fealpy.timeintegratoralg.timeline import UniformTimeLine

# scftt module
from SCFTA1BA2CLinearModel import SCFTA1BA2CLinearModel, init_value, model_options

rdir = sys.argv[1]
rhoA = init_value['C42A']
rhoB = init_value['C42B']
rhoC = init_value['C42C']
box = np.array([[4.1, 0], [0, 4.1]], dtype=np.float)
fC  = 0.14
fA2 = 0.29
fB  = 0.118
fA1 = 1-fA2-fB-fC
options = model_options(box=box, NS=64, fA1=fA1, fB=fB, fA2=fA2, fC=fC)
model = SCFTA1BA2CLinearModel(options=options)
rho = [ model.space.fourier_interpolation(rhoA), 
        model.space.fourier_interpolation(rhoB), 
        model.space.fourier_interpolation(rhoC) 
        ]
model.init_field(rho)


if True:
    for i in range(1):
        print("step:", i)
        model.compute()
        #model.test_compute_single_Q(i, rdir)
        ng = list(map(model.space.function_norm, model.grad))
        print("l2 norm of grad:", ng)
        model.update_field()

        fig = plt.figure()
        for j in range(4):
            axes = fig.add_subplot(2, 2, j+1)
            im = axes.imshow(model.w[j])
            fig.colorbar(im, ax=axes)
        fig.savefig(rdir + 'w_' + str(i) +'.png')

        fig = plt.figure()
        for j in range(3):
            axes = fig.add_subplot(1, 3, j+1)
            im = axes.imshow(model.rho[j])
            fig.colorbar(im, ax=axes)
        fig.savefig(rdir + 'rho_' + str(i) +'.png')
        plt.close()
