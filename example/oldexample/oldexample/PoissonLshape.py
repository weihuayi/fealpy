import numpy as np
import sys 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
cfont = FontProperties('SimHei')

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.functionspace.tools import function_space
from fealpy.form.Form import LaplaceSymetricForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.solver import solve
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis import L2_error, H1_semi_error
from fealpy.model.poisson_model_2d import LShapeRSinData

def lshape_mesh(r=1):
    point = np.array([
        (-1, -1),
        (0, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1)], dtype=np.float)
    cell = np.array([
        (0, 1, 3, 2),
        (2, 3, 6, 5),
        (3, 4, 7, 6)], dtype=np.int)
    tmesh = TriangleMesh(point, cell)  
    if r > 0:
        tmesh.uniform_refine(r)
    return tmesh

degree = 1
qt = 3

model = LShapeRSinData() 
mesh = lshape_mesh(r=3)
maxit = 4 
Ndof = np.zeros(maxit, dtype=np.int)
error = np.zeros(maxit, dtype=np.float)
H1error = np.zeros(maxit, dtype=np.float)
ratio = np.zeros(maxit, dtype=np.float)
for i in range(maxit):
    V = function_space(mesh, 'Lagrange', degree)
    uh = FiniteElementFunction(V)
    Ndof[i] = V.number_of_global_dofs() 
    a  = LaplaceSymetricForm(V, qt)
    L = SourceForm(V, model.source, qt)
    bc = DirichletBC(V, model.dirichlet)
    point = V.interpolation_points()
    solve(a, L, uh, dirichlet=bc, solver='direct')
    error[i] = L2_error(model.solution, uh, order=qt) 
    H1error[i] = H1_semi_error(model.gradient, uh, order=qt)
    if i < maxit-1:
        mesh.uniform_refine()

# 输出结果
ratio[1:] = error[0:-1]/error[1:]
print('Ndof:', Ndof)
print('error:', error)
print('ratio:', ratio)
print('H1semierror:', H1error)
print('ratio:', H1error[0:-1]/H1error[1:])

#fig = plt.figure()
#axes = fig.gca()
#axes.axis('tight')
#axes.axis('off')
#axes.table(cellText=[Ndof, error, ratio], rowLabels=['$Nof$', 'error', 'ratio'], loc='center')
#axes.set_title(str(degree)+u'次有限元 $L_2$ 误差表', fontproperties=cfont, y=0.6)
#plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
#plt.show()
