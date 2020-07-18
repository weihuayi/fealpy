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
from SCFTABC_BCBlendModel import SCFTABC_BCBlendModel, init_value, model_options

class SCFTABC_BCBlendModelTest():

    def __init__(self):
        pass

    def init_value(self):
        rhoA = init_value['LAM_A']
        rhoB = init_value['LAM_B']
        rhoC = init_value['LAM_C']
        print('rhoA:', rhoA)
        print('rhoB:', rhoB)
        print('rhoC:', rhoC)
        box = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=np.float)
        NS = 64
        space = FourierSpace(box,  NS)
        rhoA = space.fourier_interpolation(rhoA)
        rhoB = space.fourier_interpolation(rhoB)
        rhoC = space.fourier_interpolation(rhoC)

        print('rhoA:', rhoA)
        print('rhoB:', rhoB)
        print('rhoC:', rhoC)


    def run(self, rdir):
        rhoA = init_value['LAM_A']
        rhoB = init_value['LAM_B']
        rhoC = init_value['LAM_C']
        #rhoB = init_value['LAM']
        #rhoA = (
        #        np.array([[0, 0], [0, 0]], dtype=np.int),
        #        np.array([0, 0], dtype=np.float)
        #        )
        #rhoC = (
        #        np.array([[0, 0], [0, 0]], dtype=np.int),
        #        np.array([0, 0], dtype=np.float)
        #        )
        #box = np.array([[8 , 0, 0], [0, 8, 0], [0, 0, 8]], dtype=np.float)
        box = np.array([[4, 0], [0, 4]], dtype=np.float)
        fA  = 0.2
        fB1 = 0.4
        fC1 = 1-fA-fB1
        fBC = 0.2
        fB2 = 0.5
        fC2 = 1-fB2
        nABC = 1
        nBC = 1
        chiAB = 24
        chiAC = 24
        chiBC = 24
        options = model_options(box=box, NS=64, fA=fA, fB1=fB1, fC1=fC1, fBC=fBC, fB2=fB2, fC2=fC2, chiAB = chiAB, chiAC = chiAC, chiBC = chiBC, nABCblend = nABC, nBCblend = nBC)
        model = SCFTABC_BCBlendModel(options=options)
        rho = [ model.space.fourier_interpolation(rhoA),
                model.space.fourier_interpolation(rhoB),
                model.space.fourier_interpolation(rhoC)
                ]
        model.init_field(rho)
        H = np.inf
        maxit = options['Maxit']

        if True:
            #for i in range(maxit):
            for i in range(1):
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


test = SCFTABC_BCBlendModelTest()
if sys.argv[1] == "init_value":
    test.init_value()
elif sys.argv[1] == "run":
    test.run(rdir='./results/')


