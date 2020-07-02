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
from SCFTAB_CBlendModel import SCFTAB_CBlendModel, init_value, model_options

class SCFTAB_CBlendModelTest():

    def __init__(self):
        pass

    def init_value(self):
        rhoB = init_value['C42A']
        rhoC = init_value['C42B']
        rhoA = init_value['C42C']
        print('rhoA:', rhoA)
        print('rhoB:', rhoB)
        print('rhoC:', rhoC)
        box = np.array([[4, 0], [0, 4]], dtype=np.float)
        NS = 8
        space = FourierSpace(box,  NS)
        rhoA = space.fourier_interpolation(rhoA)
        rhoB = space.fourier_interpolation(rhoB)
        rhoC = space.fourier_interpolation(rhoC)

        print('rhoA:', rhoA)
        print('rhoB:', rhoB)
        print('rhoC:', rhoC)


    def run(self, rdir):
        rhoB = init_value['C42A']
        rhoC = init_value['C42B']
        rhoA = init_value['C42C']
        #rhoB = init_value['LAM']
        #rhoA = (
        #        np.array([[0, 0], [0, 0]], dtype=np.int),
        #        np.array([0, 0], dtype=np.float)
        #        )
        #rhoC = (
        #        np.array([[0, 0], [0, 0]], dtype=np.int),
        #        np.array([0, 0], dtype=np.float)
        #        )
        box = np.array([[4 , 0], [0, 4]], dtype=np.float)
        fC  = 0.4
        fA  = 0.6
        fB  = 0.4
        nAB = 1
        nC = 1
        options = model_options(box=box, NS=64, fA=fA, fB=fB, fC=fC, nABblend = nAB, nCblend = nC)
        model = SCFTAB_CBlendModel(options=options)
        rho = [ model.space.fourier_interpolation(rhoA),
                model.space.fourier_interpolation(rhoB),
                model.space.fourier_interpolation(rhoC)
                ]
        model.init_field(rho)
        H = np.inf
        maxit = options['Maxit']

        if True:
            for i in range(maxit):
           # for i in range(1):
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


test = SCFTAB_CBlendModelTest()
if sys.argv[1] == "init_value":
    test.init_value()
elif sys.argv[1] == "run":
    test.run(rdir='./results/')


