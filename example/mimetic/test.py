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
def solution(p, index=None):
    x = p[...,0]
    y = p[...,1]
    val = np.sin(np.pi*x) * np.sin(np.pi*y)
    #val = np.sin(2*np.pi*x) * np.sin(np.pi*y)
    return val

@cartesian
def Dirchlet(p, index=None):
    return solution(p)

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
n=10
mesh = PolygonMesh.from_unit_square(nx=n,ny=n)

solver = Mimetic(mesh)
EDdof = mesh.ds.boundary_edge_index()
div_operate = solver.div_operate()
M_c = solver.M_c()
M_f = solver.M_f()
print("M_f:", M_f.shape, "\n", M_f)
b = solver.source(source, EDdof, Dirchlet)
A10 = -M_c@div_operate
NC = mesh.number_of_cells()
A = np.bmat([[M_f, A10.T], [A10, np.zeros((NC,NC))]])

p = mesh.integral(solution,q=5,celltype=True)/mesh.entity_measure('cell')

x = np.linalg.solve(A,b)

ph = x[-NC:]
error = p-ph
print("最大误差是:", np.max(np.abs(error)))

#画图
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
fig.colorbar(surf)
plt.show()
