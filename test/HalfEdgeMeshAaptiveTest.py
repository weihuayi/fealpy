#!/usr/bin/env python3
# 
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from fealpy.mesh import HalfEdgeMesh2d,Quadtree
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh


cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
mesh = QuadrangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh)
#mesh.uniform_refine(n=4)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, showindex=True)
plt.show()
c = np.array([0.5,0.5])
r = 0.5

maxit = 6
k=0
while k < maxit:
    aopts = mesh.adaptive_options(method='numrefine', maxcoarsen=3, HB=True)
    node = mesh.node
    h = np.sqrt(mesh.cell_area())
    print('NN', mesh.number_of_nodes())

    d = np.sqrt((node[:,0]-c[0])**2+(node[:,1]-c[1])**2)
    ###要加密的点
    flag0 = np.abs(d-r) < 0.1

    ##要粗化的点
    flag1 = np.abs(d-r) > 0.1

    ###　找到点对应的单元都是需要标记的单元 
    node2cell = mesh.ds.node_to_cell()
    idx0 = np.argwhere(node2cell[flag0,:]>0)
    idx1 = np.argwhere(node2cell[flag1,:]>0)
    NC = mesh.number_of_cells()
    isrefineflag = np.zeros(NC, dtype=np.bool_)
    isrefineflag[idx0[:,1]] = 1
    iscoarsenflag = np.zeros(NC, dtype=np.bool_)
    iscoarsenflag[idx1[:,1]] = 1
    eta = np.zeros(NC)
    eta[isrefineflag] = 2
    eta[iscoarsenflag] = -1
    print(eta)

    mesh.adaptive(eta, aopts)
    cell, cellLoation = mesh.entity('cell')
    k+=1
    print('循环',k,'次***************************')
    c[0] += 0.1
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    #mesh.find_cell(axes, showindex=True)
    plt.show()

