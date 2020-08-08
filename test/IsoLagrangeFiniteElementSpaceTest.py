#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import IsoLagrangeFiniteElementSpace


class IsoLagrangeFiniteElementSpaceTest():

    def __init__(self):
        pass


    def interpolation(self, p=2, n=1, fname='surface.vtu'):
        from fealpy.geometry import SphereSurface
        from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData

        pde = SphereSinSinSinData()
        surface = pde.domain()
        mesh = pde.init_mesh(n=n)

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        space = IsoLagrangeFiniteElementSpace(mesh, p=p)

        uI = space.interpolation(pde.solution)
        mesh.nodedata['uI'] = uI[:]
        error0 = space.integralalg.error(pde.solution, uI.value) 
        error1 = space.integralalg.error(pde.gradient, uI.grad_value) 
        print(error0, error1)
        mesh.to_vtk(fname=fname)

    def surface_poisson(self, p=2, fname='surface.vtu'):
        pass

test = IsoLagrangeFiniteElementSpaceTest()

if sys.argv[1] == 'interpolation':
    p = int(sys.argv[2])
    n = int(sys.argv[3])
    fname = sys.argv[4]
    test.interpolation(p=p, n=n, fname=fname)
