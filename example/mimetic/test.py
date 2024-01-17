#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2024年01月15日 星期一 15时30分22秒
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.fem.scalar_mass_integrator import ScalarMassIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fdm.mimetic_solver import Mimetic
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from fealpy.decorator import cartesian
from scipy.interpolate import griddata
@cartesian
def source(p, index=None):
    x = p[...,0]
    y = p[...,1]
    val = 2*np.pi*np.pi*np.sin(np.pi*x) * np.sin(np.pi*y)
    return val

@cartesian
def Dirchlet(p):
    x = p[...,0]
    y = p[...,1]
    val = 0
    return val

@cartesian
def solution(p, index=None):
    x = p[...,0]
    y = p[...,1]
    val = np.sin(np.pi*x) * np.sin(np.pi*y)
    return val

@cartesian
def gradient_u(p, index=None):
    x = p[...,0]
    y = p[...,1]
    value = np.zeros_like(p)
    value[...,0] = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    value[...,1] = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    return value

node = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
                 [0.5, 0.0], [0.5, 0.5], [0.5, 1.0],
                 [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=np.float64)

cell = np.array([0, 3, 4, 1, 3, 6, 7, 4, 1, 4, 5, 2, 4, 7, 8, 4, 8, 5], dtype=np.int_)
cellLocation = np.array([0, 4, 8, 12, 15, 18], dtype=np.int_)
#mesh = PolygonMesh(node=node, cell=cell, cellLocation=cellLocation)
n=20
mesh = PolygonMesh.from_unit_square(nx=n,ny=n)
NC = mesh.number_of_cells()
NE = mesh.number_of_edges()
solver = Mimetic(mesh)

EDdof = mesh.ds.boundary_edge_index()
div_operate = solver.div_operate()
M_c = solver.M_c()
M_f = solver.M_f()
b = solver.source(source, EDdof, Dirchlet)[NE:]

'''
qf = mesh.integrator(5,etype='edge')
bcs,ws = qf.get_quadrature_points_and_weights()
carpoint = mesh.edge_bc_to_point(bcs)
edge_measure = mesh.entity_measure(etype=1)
cell_measure = mesh.entity_measure(etype=2)
normal = mesh.edge_unit_normal()

u = np.einsum('i, ijk, jk-> j', ws, gradient_u(carpoint), normal)
p = mesh.integral(solution,q=5,celltype=True)/cell_measure
A10 = -M_c@div_operate
b = -M_c@b



print("div_operate 验算", np.max(np.abs(A10@u - b)))
print("M_f 验算", np.max(np.abs(A10.T@p - M_f@u)))
'''


b = solver.source(source, EDdof, Dirchlet)
A10 = -M_c@div_operate
A = np.bmat([[M_f, A10.T], [A10, np.zeros((NC,NC))]])
#print(A10)
#print('单元中点',mesh.entity_barycenter(etype=2))
#print('边中点',mesh.entity_barycenter(etype=1))
#print('单元面积',mesh.entity_measure('cell'))
#print('边长度',mesh.entity_measure('edge'))
#print('边法向',mesh.edge_unit_normal())
p = mesh.integral(solution,q=4,celltype=True)/mesh.entity_measure('cell')
#p = solution(mesh.entity_barycenter(etype=2))
Ddof = mesh.ds.boundary_cell_flag()
#A,b = solver.boundary_treatment(A,b, Dirchlet, Ddof, so=u)
x = np.linalg.solve(A,b)
ph = x[NE:]
error = p-ph
print(np.max(np.abs(p-ph)))
cc = mesh.entity_barycenter(etype=2)
x = cc[:,0]
y = cc[:,1]
x, y = np.meshgrid(x, y)
z = griddata(cc, ph, (x, y), method='linear')

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图像
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k')
#plt.scatter(cc[:,0], cc[:,1], c=error, cmap='viridis', marker='o')
fig.colorbar(surf)
#plt.colorbar()
plt.show()
'''
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, color='r', marker='o', markersize=16, fontsize=32, fontcolor='r')
mesh.find_cell(axes, showindex=True, color='b', marker='o', markersize=16, fontsize=32, fontcolor='b')
mesh.find_edge(axes, showindex=True, color='g', marker='o', markersize=16, fontsize=32, fontcolor='g')
plt.show()
'''
