#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import FourierSpace 
from fealpy.mesh import StructureQuadMesh


class FourierSpaceTest():
    def __init__(self):
        pass

    def linear_equation_fft_solver_1d_test(self, N):
        def u(p):
            x = p[0]
            return np.sin(x) 

        def f(p):
            x = p[0]
            return 2*np.sin(x) 

        box = np.array([[2*np.pi]]) 
        mesh = FourierSpace(box, N)
        U = mesh.linear_equation_fft_solver(f)
        error = mesh.error(u, U)
        print(error)

    def linear_equation_fft_solver_2d_test(self, N):
        def u(p):
            x = p[0]
            y = p[1]
            return np.sin(x)*np.sin(y)

        def f(p):
            x = p[0]
            y = p[1]
            return 3*np.sin(x)*np.sin(y) 

        box = np.array([
            [2*np.pi, 0],
            [0, 2*np.pi]]) 
        mesh = FourierSpace(box, 6)
        xi = mesh.reciprocal_lattice(sparse=False)
        U = mesh.linear_equation_fft_solver(f)
        error = mesh.error(u, U)
        print(error)

    def linear_equation_fft_solver_3d_test(self, N):
        def u(p):
            x = p[0]
            y = p[1]
            z = p[2]
            return np.sin(x)*np.sin(y)*np.sin(z)

        def f(p):
            x = p[0]
            y = p[1]
            z = p[2]
            return 4*np.sin(x)*np.sin(y)*np.sin(z)

        box = np.array([
            [2*np.pi, 0, 0],
            [0, 2*np.pi, 0],
            [0, 0, 2*np.pi]]) 
        mesh = FourierSpace(box, 6)
        xi = mesh.reciprocal_lattice(sparse=False)
        U = mesh.linear_equation_fft_solver(f)
        error = mesh.error(u, U)
        print(error)

    def box_test(self, N):

        box = np.array([
            [2*np.pi, 0],
            [0, 2*np.pi]])
        mesh = FourierSpace(box, N)
        node = mesh.node
        print(node)
        


test = FourierSpaceTest()

if True:
    test.linear_equation_fft_solver_1d_test(10)

if True:
    test.linear_equation_fft_solver_2d_test(6)

if True:
    test.linear_equation_fft_solver_3d_test(6)

if True:
    test.box_test(4)



if False:
    qmesh = StructureQuadMesh(box, N, N)
    mi = qmesh.multi_index()
    fig = plt.figure()
    axes = fig.gca()
    qmesh.add_plot(axes)
    qmesh.find_node(axes, showindex=True)
    plt.show()
