import sys
import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.simple_mesh_generator import unitcircledomainmesh
from fealpy.mesh import PolygonMesh

def tri_to_polygonmesh(mesh, n):
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    edge = mesh.entity('edge')

    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()

    isBDEdge = mesh.ds.boundary_edge_flag()
    w = np.zeros((n, 2), dtype=mesh.ftype)
    w[:, 1] = np.linspace(1/(n+1), 1, n, endpoint=False) 
    w[:, 0] = 1 - w[:, 1]
    bc = np.einsum('ij, ...jk->...ik', w, node[edge[isBDEdge, :]])
    l = np.sqrt(np.sum(bc**2, axis=-1))
    bc /= l[..., np.newaxis]

    NN = mesh.number_of_nodes()
    NB= bc.shape[0]
    idx0 = np.arange(NN, NN+NB*n).reshape(NB, n)

    idxmap = np.zeros(NE, dtype=mesh.itype)
    idxmap[isBDEdge] = range(NB)

    NV = 3*np.ones(NC, dtype=mesh.itype)
    
    edge2cell = mesh.ds.edge_to_cell()
    np.add.at(NV, edge2cell[isBDEdge, 0], n)
    
    pcellLocation = np.add.accumulate(np.r_[0, NV])
    pcell = np.zeros(pcellLocation[-1], dtype=mesh.itype)
    isBDCell = mesh.ds.boundary_cell_flag()

    pcell[pcellLocation[0:-1][~isBDCell]+0] = cell[~isBDCell, 0]
    pcell[pcellLocation[0:-1][~isBDCell]+1] = cell[~isBDCell, 1]
    pcell[pcellLocation[0:-1][~isBDCell]+2] = cell[~isBDCell, 2]

    
    cellidx, = np.nonzero(isBDCell)
    start = pcellLocation[cellidx]
    pcell[start] = cell[cellidx, 0]

    cell2edge = mesh.ds.cell_to_edge()
    flag = isBDEdge[cell2edge[cellidx, 2]]
    idx1 = start[flag].reshape(-1, 1) + np.arange(1, n+1)
    pcell[idx1] = idx0[idxmap[cell2edge[cellidx[flag], 2]], :] 

    start[flag] += n+1
    start[~flag] += 1
    pcell[start] = cell[cellidx, 1]

    flag = isBDEdge[cell2edge[cellidx, 0]]
    idx1 = start[flag].reshape(-1, 1) + np.arange(1, n+1)
    pcell[idx1] = idx0[idxmap[cell2edge[cellidx[flag], 0]], :]

    start[flag] += n+1
    start[~flag] += 1
    pcell[start] = cell[cellidx, 2]

    flag = isBDEdge[cell2edge[cellidx, 1]]
    idx1 = start[flag].reshape(-1, 1) + np.arange(1, n+1)
    pcell[idx1] = idx0[idxmap[cell2edge[cellidx[flag], 1]], :]

    pnode = np.r_[node, bc.reshape(-1, 2)]
    return PolygonMesh(pnode, pcell, pcellLocation)






mesh = unitcircledomainmesh(0.4, meshtype='tri')
pmesh = tri_to_polygonmesh(mesh, 2)

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
#mesh.add_plot(axes)
print(pmesh.ds.cell)
print(pmesh.ds.cellLocation)
pmesh.find_node(axes, showindex=True)
pmesh.find_edge(axes, showindex=True)
pmesh.find_cell(axes, showindex=True)
plt.show()
