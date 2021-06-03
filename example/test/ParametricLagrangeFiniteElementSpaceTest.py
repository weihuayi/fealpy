#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace


class ParametricLagrangeFiniteElementSpaceTest():

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
        space = ParametricLagrangeFiniteElementSpace(mesh, p=p)

        uI = space.interpolation(pde.solution)
        mesh.nodedata['uI'] = uI[:]
        error0 = space.integralalg.error(pde.solution, uI.value) 
        error1 = space.integralalg.error(pde.gradient, uI.grad_value) 
        print(error0, error1)
        mesh.to_vtk(fname=fname)

    def plane_quad_interpolation(self, p=2, fname='plane.vtu'):
        from fealpy.pde.poisson_2d import CosCosData 
        from fealpy.mesh import LagrangeQuadrangleMesh
        pde = CosCosData()
        node = np.array([
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1)], dtype=np.float64)
        cell = np.array([
            (0, 1, 2, 3)], dtype=np.int_)

        mesh = LagrangeQuadrangleMesh(node, cell, p=p)

        for i in range(4): 
            space = ParametricLagrangeFiniteElementSpace(mesh, p=p)
            uI = space.interpolation(pde.solution)
            mesh.nodedata['uI'] = uI[:]
            error0 = space.integralalg.error(pde.solution, uI.value) 
            error1 = space.integralalg.error(pde.gradient, uI.grad_value) 
            print(error0, error1)
            if i < 3:
                mesh.uniform_refine()

        mesh.to_vtk(fname=fname)

test = ParametricLagrangeFiniteElementSpaceTest()

if sys.argv[1] == 'SI':
    p = int(sys.argv[2])
    n = int(sys.argv[3])
    fname = sys.argv[4]
    test.interpolation(p=p, n=n, fname=fname)
elif sys.argv[1] == 'PQI':
    p = int(sys.argv[2])
    fname = sys.argv[3]
    test.plane_quad_interpolation(p=p, fname=fname)
