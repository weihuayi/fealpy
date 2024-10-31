import numpy as np
import sys 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
cfont = FontProperties('SimHei')

from fealpy.mesh.simple_mesh_generator import rectangledomainmesh, triangle
from fealpy.functionspace.tools import function_space
from fealpy.form.Form import LaplaceSymetricForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.solver import solve
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis import L2_error
from fealpy.model.poisson_model_2d import CosCosData

degree = int(sys.argv[1]) 
qt = int(sys.argv[2])  
n = int(sys.argv[3])

box = [0, 1, 0, 1]

model = CosCosData()
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
maxit = 4 
Ndof = np.zeros(maxit, dtype=np.int)
error = np.zeros(maxit, dtype=np.float)
ratio = np.zeros(maxit, dtype=np.float)
for i in range(maxit):
    V = function_space(mesh, 'Lagrange', degree)
    uh = FiniteElementFunction(V)
    Ndof[i] = V.number_of_global_dofs() 
    a  = LaplaceSymetricForm(V, qt)
    L = SourceForm(V, model.source, qt)
    bc = DirichletBC(V, model.dirichlet, model.is_boundary)
    point = V.interpolation_points()
    solve(a, L, uh, dirichlet=bc, solver='direct')
    error[i] = L2_error(model.solution, uh, order=qt) 
    # error[i] = np.sqrt(np.sum((uh - model.solution(point))**2)/Ndof[i])
    if i < maxit-1:
        mesh.uniform_refine()

# 输出结果
ratio[1:] = error[0:-1]/error[1:]
print('Ndof:', Ndof)
print('error:', error)
print('ratio:', ratio)

#fig = plt.figure()
#axes = fig.gca()
#axes.axis('tight')
#axes.axis('off')
#axes.table(cellText=[Ndof, error, ratio], rowLabels=['$Nof$', 'error', 'ratio'], loc='center')
#axes.set_title(str(degree)+u'次有限元 $L_2$ 误差表', fontproperties=cfont, y=0.6)
#plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
#plt.show()
