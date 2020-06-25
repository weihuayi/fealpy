#!/usr/bin/env python3
# 
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.pde.poisson_2d import CosCosData, X2Y2Data
from fealpy.functionspace.Function import Function


class FunctionTest():

    def __init__(self):
        pass

    def interpolation(self, n=0, p=0): 
        pde = CosCosData()
        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        uI = space.interpolation(pde.flux)
        print(space.basis.coordtype)
        u = Function(space)
        u[:] = uI

        qf = mesh.integrator(3, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        error = space.integralalg.L2_error(pde.source, u.div_value)
        print('error:', error)
        print(dir(u))

test = FunctionTest()
if sys.argv[1] == 'interpolation':
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.interpolation(n=n, p=p)
    
