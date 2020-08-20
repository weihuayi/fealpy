#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

from problem import quad_model, plane_model, converge_model
from SCFTVEMModel import scftmodel_options
from fealpy.opt.saddleoptalg import SteepestDescentAlg
from fealpy.quadrature import TriangleQuadrature


order = int(sys.argv[1])
n = int(sys.argv[2])

moptions = scftmodel_options(
        nspecies= 2,
        nblend = 1,
        nblock = 2,
        ndeg = 100,
        fA = 0.2,
        chiAB = 0.25,
        dim = 2,
        T0 = 20,
        T1 = 80,
        nupdate = 0,
        order = order)

mesh = complex_mesh(r=20, filename = sys.argv[2])
options = {
        'MaxIters'          :500000,
        'MaxFunEvals'       :500000,
        'NormGradTol'       :1e-6,
        'FunValDiff'        :1e-6,
        'StepLength'        :2,
        'StepTol'           :1e-14,
        'Output'            :True
        }
optalg = SteepestDescentAlg(problem, options)
optalg.run(maxit=1)
