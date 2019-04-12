#!/usr/bin/env python3
#
"""
This script include the test code of the adaptive VEM for the simplified
friction problem.
"""

import numpy as np
from fealpy.pde.sfc_2d import SFCModelData2
from fealpy.vem.SFCVEMModel2d import SFCVEMModel2d

pde = SFCModelData2()
qtree = pde.init_mesh(n=2, meshtype='quadtree')
pmesh = qtree.to_pmesh()

vem = SFCVEMModel2d(pde, pmesh, p=1, q=4)
vem.solve(rho=0.1, maxit=40000)
