#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

from problem import plane_model
from PRISMSCFTFEMModel import pscftmodel_options
from fealpy.opt.saddleoptalg import SteepestDescentAlg


order = int(sys.argv[1])
n = int(sys.argv[2])

moptions = pscftmodel_options(
        nspecies= 2,
        nblend = 1,
        nblock = 2,
        ndeg = 100,
        fA = 0.2,
        chiAB = 0.25,
        dim = 3,
        T0 = 20,
        T1 = 80,
        nupdate = 1,
        order = order)

problem = plane_model(fieldstype=1, n=n, options=moptions)
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
optalg.run()
