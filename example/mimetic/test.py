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
from mumps import DMumpsContext
from scipy.sparse import csr_matrix
@cartesian
def source(p, index=None):
    x = p[...,0]
    y = p[...,1]
    val = 2*np.pi*np.pi*np.sin(np.pi*x) * np.sin(np.pi*y)
    #val = 5 * np.pi**2 *np.sin(2*np.pi*x) * np.sin(np.pi*y)
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
    #val = np.sin(2*np.pi*x) * np.sin(np.pi*y)
    return val

@cartesian
def gradient_u(p, index=None):
    x = p[...,0]
    y = p[...,1]
    value = np.zeros_like(p)
    value[...,0] = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    value[...,1] = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    return value

@cartesian
def div_u(p, index=None):
    x = p[...,0]
    y = p[...,1]
    value0 = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    value1 = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    value = value0+value1
    return value

node = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
                 [0.5, 0.0], [0.5, 0.5], [0.5, 1.0],
                 [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=np.float64)

cell = np.array([0, 3, 4, 1, 3, 6, 7, 4, 1, 4, 5, 2, 4, 7, 8, 4, 8, 5], dtype=np.int_)
cellLocation = np.array([0, 4, 8, 12, 15, 18], dtype=np.int_)
#mesh = PolygonMesh(node=node, cell=cell, cellLocation=cellLocation)
n=10
mesh = PolygonMesh.from_unit_square(nx=n,ny=n)
NC = mesh.number_of_cells()
NE = mesh.number_of_edges()
solver = Mimetic(mesh)

EDdof = mesh.ds.boundary_edge_index()
div_operate = solver.div_operate()
M_c = solver.M_c()
M_f = solver.M_f()
print("M_f:", M_f.shape, "\n", M_f)
b = solver.source(source, EDdof, Dirchlet)
A10 = -M_c@div_operate
A = np.bmat([[M_f, A10.T], [A10, np.zeros((NC,NC))]])


qf = mesh.integrator(5,etype='edge')
bcs,ws = qf.get_quadrature_points_and_weights()
carpoint = mesh.edge_bc_to_point(bcs)
edge_measure = mesh.entity_measure(etype=1)
cell_measure = mesh.entity_measure(etype=2)
normal = mesh.edge_unit_normal()

'''
u = np.einsum('i, ijk, jk-> j', ws, gradient_u(carpoint), normal)
p = mesh.integral(solution,q=5,celltype=True)
divp = mesh.integral(div_u,q=5,celltype=True)
b = mesh.integral(source,q=5,celltype=True)
print(np.max(np.abs(M_c@div_operate@u-M_c@b)))
x = np.hstack((u,p))
error = np.einsum('ij,j->i',A,x) - b
print(np.sqrt(np.sum(error**2)/(NC+NE)))
'''
#print("div_operate 验算", np.max(np.abs(A10@u - b)))
#print("M_f 验算", np.max(np.abs(A10.T@p - M_f@u)))

#print('单元中点',mesh.entity_barycenter(etype=2))
#print('边中点',mesh.entity_barycenter(etype=1))
#print('单元面积',mesh.entity_measure('cell'))
#print('边长度',mesh.entity_measure('edge'))
#print('边法向',mesh.edge_unit_normal())
#p = solution(mesh.entity_barycenter(etype=2))

Ddof = mesh.ds.boundary_cell_flag()
p = mesh.integral(solution,q=5,celltype=True)/mesh.entity_measure('cell')
A,b = solver.boundary_treatment(A,b, Dirchlet, Ddof, so=p)

x = np.linalg.solve(A,b)
'''
A = csr_matrix(A)
ctx = DMumpsContext()
ctx.set_silent()
x = b.copy()
ctx.set_centralized_sparse(A)
ctx.set_rhs(x)
ctx.run(job=6)
'''
ph = x[-NC:]

error = p-ph
print(np.max(np.abs(error[Ddof])))
print(np.max(np.abs(error)))
cc = mesh.entity_barycenter(etype=2)
x = cc[:,0]
y = cc[:,1]
x, y = np.meshgrid(x, y)
z = griddata(cc, error, (x, y), method='linear')

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
#plt.show()
'''
