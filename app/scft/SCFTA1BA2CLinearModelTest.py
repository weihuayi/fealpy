#!/usr/bin/env python3
#

import sys
import math
import time

import numpy as np
import matplotlib.pyplot as plt

# fealpy module
from fealpy.functionspace import FourierSpace
from fealpy.timeintegratoralg.timeline import UniformTimeLine

# scftt module
from SCFTA1BA2CLinearModel import SCFTA1BA2CLinearModel, init_value, model_options

class SCFTA1BA2CLinearModelTest():

    def __init__(self):
        pass

    def init_value(self):
        rhoA = init_value['C42A']
        rhoB = init_value['C42B']
        rhoC = init_value['C42C']
        print('rhoA:', rhoA)
        print('rhoB:', rhoB)
        print('rhoC:', rhoC)
        box = np.array([[4.1, 0], [0, 4.1]], dtype=np.float)
        NS = 8
        space = FourierSpace(box,  NS)
        rhoA = space.fourier_interpolation(rhoA)
        rhoB = space.fourier_interpolation(rhoB)
        rhoC = space.fourier_interpolation(rhoC)

        print('rhoA:', rhoA)
        print('rhoB:', rhoB)
        print('rhoC:', rhoC)


    def run(self, rdir):
        rhoA = init_value['C42A']
        rhoB = init_value['C42B']
        rhoC = init_value['C42C']
        #rhoB = init_value['LAM']
        #rhoA = (
        #        np.array([[0, 0], [0, 0]], dtype=np.int),
        #        np.array([0, 0], dtype=np.float)
        #        )
        #rhoC = (
        #        np.array([[0, 0], [0, 0]], dtype=np.int),
        #        np.array([0, 0], dtype=np.float)
        #        )
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
        H = np.inf
        maxit = options['Maxit']

        if True:
            for i in range(maxit):
                print("step:", i)
                model.compute()
                print('H:',H)
                H_diff = np.abs(H - model.H)
                print('Hmodel:', model.H)
                print('Hdiff:',H_diff)
                H = model.H
                ng = list(map(model.space.function_norm, model.grad))
                print("l2 norm of grad:", ng)
                if H_diff < options['tol']:
                    break
            model.save_data()


test = SCFTA1BA2CLinearModelTest()
if sys.argv[1] == "init_value":
    test.init_value()
elif sys.argv[1] == "run":
    start =time.clock()
    test.run(rdir='./results/')
    end =time.clock()
    print('Running time: %s Seconds'%(end-start))


