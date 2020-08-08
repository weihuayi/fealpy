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


    def interpolation(self, p=2, fname='surface.vtu'):
        from fealpy.geometry import SphereSurface

        @cartesian
        def u(p):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            pi = np.pi
            return np.sin(4*pi*x)*np.sin(4*pi*y)*np.sin(4*pi*z)
        surface = SphereSurface()
        mesh = surface.init_mesh()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        space = IsoLagrangeFiniteElementSpace(mesh, p=p)

        uI = space.interpolation(u)
        mesh.nodedata['uI'] = uI[:]
        error = space.integralalg.error(u, uI) 
        print(error)
        mesh.to_vtk(fname=fname)

test = IsoLagrangeFiniteElementSpaceTest()

if sys.argv[1] == 'interpolation':
    p = int(sys.argv[2])
    fname = sys.argv[3]
    test.interpolation(p=p)
