#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_wang.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 07 May 2024 10:56:04 AM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.decorator import cartesian
import matplotlib.pyplot as plt
@cartesian
def source(p, index=None):
    x = p[...,0]
    y = p[...,1]
    #val = 5*np.pi*np.pi*np.cos(2*np.pi*x)*np.cos(np.pi*y)
    #val = 5*np.pi*np.pi*np.sin(2*np.pi*x)*np.sin(np.pi*y)
    val = np.zeros_like(x)
    #val = 2*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)
    return val


@cartesian
def solution(p, index=None):
    x = p[...,0]
    y = p[...,1]
    #val = np.sin(2*np.pi*x)*np.sin(np.pi*y)
    val = np.exp(x)*np.sin(y)
    #val = np.cos(2*np.pi*x)*np.cos(np.pi*y)
    #val = np.sin(np.pi*x)*np.sin(np.pi*y)
    return val

ns = 16
mesh = PolygonMesh.from_unit_square(nx=ns, ny=ns)

cell = mesh.entity('cell')
node = mesh.entity('node')
edge = mesh.entity('edge')
edge_measure = mesh.entity_measure('edge')
cell_measure = mesh.entity_measure('cell')
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
cell2edge = mesh.ds.cell_to_edge()
x_c = mesh.entity_barycenter(etype=2)
x_e = mesh.entity_barycenter(etype=1)
unit_tangent = mesh.edge_unit_tangent()
unit_normal = mesh.edge_unit_normal()
M_E = np.zeros((NE,NE))
M_V = np.zeros((NN, NN))
flag = np.where(mesh.ds.cell_to_edge_sign().toarray(), 1, -1)

for i in range(NC):
    # ME
    LNE = len(cell2edge[i])
    cell_0 = np.append(cell[i],cell[i][0])
    cell_tangent = node[cell_0[1:]] - node[cell_0[:-1]]
    cell_unit_tangent_c = -cell_tangent/ np.linalg.norm(cell_tangent,axis=1,keepdims=True)
    cell_unit_tangent = unit_tangent[cell2edge[i]]
    beta = np.sum(cell_unit_tangent*cell_unit_tangent_c,axis=1)
    
    pi = np.zeros((LNE,2))
    pi[:,0] = (x_e[cell2edge[i]]-x_c[i])[:,1]
    pi[:,1] = -(x_e[cell2edge[i]]-x_c[i])[:,0]
    R = np.einsum('i,ij,i->ij',beta,pi,edge_measure[cell2edge[i]])
    
    N = cell_unit_tangent
    
    M_consistency = R @ np.linalg.inv(R.T @ N) @ R.T
    #M_consistency = R@R.T/cell_measure[i]
    M_stability = np.trace(M_consistency)*(np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)/(LNE*LNE)
    #M_stability = cell_measure[i]*(np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
    M = M_consistency + M_stability

    indexi, indexj = np.meshgrid(cell2edge[i], cell2edge[i])
    M_E[indexi, indexj] += M
    
    #MV
    LNV = len(cell[i])
    tmp = flag[i, cell2edge[i]].reshape(-1, 1) 
    # 单位外法向量
    cell_unit_outward_normals = unit_normal[cell2edge[i]] * tmp 
    R_V = 0.5 * np.einsum('lg, lg, l -> l', cell_unit_outward_normals, \
                    x_e[cell2edge[i]]-x_c[i], edge_measure[cell2edge[i]]) 
    R_V = R_V.reshape(-1,1)
    N_V = np.ones(LNV).reshape(-1,1)
    M_V_consistency = R_V @ np.linalg.inv(R_V.T @ N_V) @ R_V.T
    #M_V_consistency = R_V @ R_V.T/cell_measure[i]
    M_V_stability = np.trace(M_V_consistency)*(np.eye(LNV) - N_V @ np.linalg.inv(N_V.T @ N_V) @ N_V.T)/(LNV*LNV)
    #M_V_stability = cell_measure[i]*(np.eye(LNV) - N_V @ np.linalg.inv(N_V.T @ N_V) @ N_V.T)
    MV = M_V_consistency + M_V_stability

    indexi, indexj = np.meshgrid(cell[i], cell[i])
    M_V[indexi, indexj] += MV


gradh = np.zeros((NE, NN))
for i in range(NE):
    gradh[i, edge[i, 0]] = -1 / edge_measure[i]
    gradh[i, edge[i, 1]] = 1 / edge_measure[i]
    
A = gradh.T@M_E@gradh
f = source(node)
b = M_V@f


eDdof = mesh.ds.boundary_edge_index()
nDdof = mesh.entity('edge')[eDdof][:, 0]
b[nDdof] = solution(node[nDdof])

bdIdx = np.zeros(A.shape[0], dtype=np.int_)
bdIdx[nDdof.flat] = 1
from scipy.sparse import spdiags
D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0]).toarray()
D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0]).toarray()
A = D0 @ A + D1
uh = np.linalg.solve(A,b)
print(np.sum(np.abs(A)))
print(np.sum(np.abs(b)))
print(np.sum(np.abs(uh)))
uso = solution(node)
#print(np.sum(np.abs(uh-uso))/NN)
print(np.max(np.abs(uh-uso)))


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes,showindex=True)
mesh.find_node(axes,showindex=True)
mesh.find_edge(axes,showindex=True)
#plt.show()
