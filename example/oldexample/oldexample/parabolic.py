import sys

import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve

import matplotlib.pyplot as plt

from fealpy.model.parabolic_model_2d import  SinCosExpData
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.functionspace.tools import function_space
from fealpy.form.Form import LaplaceSymetricForm, MassForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.erroranalysis import L2_error
from fealpy.functionspace.function import FiniteElementFunction

model = SinCosExpData()
box = [0, 1, 0, 1]
n = 10 
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
V = function_space(mesh, 'Lagrange', 1)
Ndof = V.number_of_global_dofs()
A  = LaplaceSymetricForm(V, 3).get_matrix()
M = MassForm(V, 3).get_matrix()
b = SourceForm(V, model.source, 1).get_vector()
BC = DirichletBC(V, model.dirichlet, model.is_dirichlet_boundary)

T0 = 0.0
T1 = 1
N = 400 
dt = (T1 - T0)/N
print(dt)

uh = FiniteElementFunction(V)
uh[:] = model.init_value(V.interpolation_points())

uht = [uh]
MD = BC.apply_on_matrix(M)
for i in range(1, N+1):
    t = T0 + i*dt

    AD = M + dt/16*A
    F = M@uht[i-1] + dt*b
    AD, F = BC.apply(AD, F)
    uh = FiniteElementFunction(V)
    uh[:] = spsolve(AD, F) 

#    F = dt*(b - 1/16*A@uht[i-1]) + M@uht[i-1]
#    F = BC.apply_on_source_vector(F, M)
#    uh[:] = spsolve(MD, F)
    uht.append(uh)
    e = L2_error(lambda p: model.solution(p, t), uh)
    print(e)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
