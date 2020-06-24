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
        u = Function(space)
        print(dir(u))

test = FunctionTest()
if sys.argv[1] == 'interpolation':
    test.interpolation()
    
