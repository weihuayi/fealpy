#!/usr/bin/env python3
# 
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace.Function_1 import Function


class FunctionTest():

    def __init__(self):
        pass

    def print_method(self, n=0, p=0): 
        pde = CosCosData()
        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = LagrangeFiniteElementSpace(mesh, p=p)

        array = space.array(dim=2)

        uI = Function(space, array=array)
        qf = mesh.integrator(3, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        print("__call__:", uI(bcs))
        print("value:", uI.value(bcs))
        print(uI[1:2])
        print(uI.index(0))
        print(uI.__dict__)

test = FunctionTest()
if sys.argv[1] == 'method':
    test.print_method(n=1, p=1)
    
