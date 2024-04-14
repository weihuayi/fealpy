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
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.fdm.mimetic_solver import Mimetic
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

@cartesian
def source(p, index=None):
    x = p[...,0]
    y = p[...,1]
    val = 2*np.pi*np.pi*np.sin(np.pi*x) * np.sin(np.pi*y)
    return val


@cartesian
def solution(p, index=None):
    x = p[...,0]
    y = p[...,1]
    val = np.sin(np.pi*x) * np.sin(np.pi*y)
    return val

@cartesian
def Dirchlet(p, index=None):
    return solution(p)

ns=2
mesh = PolygonMesh.from_unit_square(nx=ns,ny=ns)
maxit = 6
errorType = ['$max( p - p_h)$',]
errorMatrix = np.zeros((1, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)


for i in range(maxit):
    print("The {}-th computation:".format(i))
    solver = Mimetic(mesh)
    NDof[i] = mesh.number_of_cells()
    EDdof = mesh.ds.boundary_edge_index()
    div_operate = solver.div_operate()
    M_c = solver.M_c()
    M_f = solver.M_f()

    b = solver.source(source, EDdof, Dirchlet)
    A10 = -M_c@div_operate
    NC = mesh.number_of_cells()
    A = np.bmat([[M_f, A10.T], [A10, np.zeros((NC,NC))]])

    x = np.linalg.solve(A,b)

    p = mesh.integral(solution,q=5,celltype=True)/mesh.entity_measure('cell')
    ph = x[-NC:]
    errorMatrix[0,i] = np.max(np.abs(p-ph))
    if i < maxit-1:
        ns *= 2
        mesh = PolygonMesh.from_unit_square(nx=ns,ny=ns)

showmultirate(plt, 2, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
plt.show()
