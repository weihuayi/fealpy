import numpy as np
import taichi as ti
import math

from fealpy.mesh import MeshFactory as MF
from fealpy.ti import TriangleMesh # 基于 Taichi 的三角形网格

ti.init()

@ti.func
def velocity(x: ti.f64, y: ti.f64) -> ti.f64:
    pi = math.pi
    u0 = ti.sin(pi*x)**2*ti.sin(2*pi*y)
    u1 =-ti.sin(pi*y)**2*ti.sin(2*pi*x)
    return  ti.Vector([u0, u1]) 

@ti.func
def f(x: ti.f64, y: ti.f64) -> ti.f64:
    z = x*x + y*y
    return  z 

node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)

mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

K = mesh.stiff_matrix()

F = mesh.source_vector(f)

val0 = ti.field(ti.f64, NN)
mesh.scalar_interpolation(f, val0)

u = ti.field(ti.f64, (NN, 2))
mesh.vector_interpolation(velocity, u)

C = mesh.convection_matrix(u)
print(C.toarray())

A = mesh.ti_stiff_matrix()









