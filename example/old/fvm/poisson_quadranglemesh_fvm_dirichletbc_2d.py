#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: poisson_quadranglemesh_fvm_dirichletbc_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2024年01月02日 星期二 19时20分18秒
	@bref 
	@ref 
'''  
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.mesh import QuadrangleMesh
from fealpy.fvm import ScalarDiffusionIntegrator
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

def solution(p):
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    return np.cos(pi*x)*np.cos(pi*y)
def source(p):
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    return 2*pi**2*np.cos(pi*x)*np.cos(pi*y) 
def dirichlet(p):
    return solution(p)
def is_dirichlet_boundary(p):
    eps = 1e-12
    x = p[..., 0]
    y = p[..., 1]
    return (np.abs(y-1)<eps)|(np.abs(x-1)<eps)|(np.abs(x)<eps)|(np.abs(y)<eps)

ns = 4
domain = [0,1,0,1]
mesh = QuadrangleMesh.from_box(box=domain,nx=ns,ny=ns)
maxit=6
#errorType = ['$max( u - u_h)$',]
errorType = ['$||( u - u_h)||_{l2}$',]
errorMatrix = np.zeros((1, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    NDof[i] = mesh.number_of_nodes()
    DI = ScalarDiffusionIntegrator(mesh)

    DM,Db = DI.cell_center_matrix(dirichlet, is_dirichlet_boundary)

    bb = mesh.integral(source, celltype=True)
    uh = spsolve(DM, bb+Db)

    ipoint = mesh.entity_barycenter('cell')
    u = solution(ipoint)
    
    e = u - uh 
    h = (domain[1]-domain[0])/ns
    #errorMatrix[0,i] = np.max(np.abs(e))
    errorMatrix[0,i] = np.sqrt(np.sum(h*h*e**2))
    if i < maxit-1:
        ns *= 2
        mesh = QuadrangleMesh.from_box(box=domain,nx=ns,ny=ns)
    #print('emax', np.max(np.abs(u-uh)))
    #print('eL2', np.sqrt(np.sum(h*h*e**2)))

showmultirate(plt, 3, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
plt.xlabel("Number of dofs")
plt.ylabel("l2 Error")
plt.show()
'''
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(nx, ny)
Y = yy.reshape(ny, ny)
Z = uh.reshape(nx, ny)
ax1.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()
'''

