import numpy as np
from scipy.spatial import Delaunay

from ..geometry import find_cut_point
from .TriangleMesh import TriangleMesh
from .StructureQuadMesh import StructureQuadMesh

def adaptive(mesh, interface, hmax):
    """
    @brief 生成自适应的界面拟合网格 
    """

    mesh.bisect_interface_cell_with_curvature(interface, hmax)

    NN = mesh.number_of_nodes()
    cell = mesh.entity('cell')
    edge = mesh.entity('edge')
    edge2cell = mesh.ds.edge_to_cell()
    isInterfaceCell = mesh.mark_interface_cell(phi)
    isInterfaceNode = np.zeros(NN, dtype=np.bool_)
    isInterfaceNode[cell[isInterfaceCell]] = True

    isEdge = edge2cell[:, 0] != edge2cell[:, 1]
    isEdge = isEdge & ((isInterfaceNode[edge].sum(axis=1) > 0) |
            (isInterfaceNode[cell[edge2cell[:, 0:2], 0]].sum(axis=1)> 0))
    isEdge = isEdge & (edge2cell[:, 3:].sum(axis=1) == 0)

    isEdge = isEdge & (~cellType[edge2cell[:, 0]])

    NC = mesh.number_of_cells()
    isMark = np.zeros(NC, dtype=np.bool_)
    isMark[edge2cell[isEdge, 0:2]] = True

    mesh.bisect(isMark)
    node = mesh.entity('node')
    phi = np.append(phi, interface(node[NN:]))
    NN = mesh.number_of_nodes()

    cell = mesh.entity('cell')
    v = node[cell[:, 2]] - node[cell[:, 1]]
    cellType = (np.abs(v[:, 0]) > 0.0) & (np.abs(v[:, 1]) > 0.0) # TODO: 0.0-->eps


    # Step 4: move some interface nodes onto the interface

    edge = mesh.entity('edge')
    edge2cell = mesh.ds.edge_to_cell()

    isInterfaceCell = mesh.mark_interface_cell(phi)
    isInterfaceEdge = isInterfaceCell[edge2cell[:, 0:2]].sum(axis=1) == 2 

    isSpecial0 = cellType[edge2cell[:, 0]] & (~cellType[edge2cell[:, 1]]) & edge2cell[:, 2] == 0 & isInterfaceEdge
    
    isSpecial1 = (~cellType[edge2cell[:, 0]]) & cellType[edge2cell[:, 1]] & edge2cell[:, 2] == 0 & isInterfaceEdge

    isSpecial = isSpecial0 | isSpecial1

    isShortEdge = (edge2cell[:, 2] != 0) & (edge2cell[:, 3] == 0) & (~isSpecial) 
    A = node[edge[isShortEdge, 0]]
    B = node[edge[isShortEdge, 1]]
    h = np.sqrt(np.sum((A - B)**2, axis=1))
    M = find_cut_point(interface, A, B)
    return  M 


def uniform_interface_fitted_mesh2d(interface, mesh):
    """
    @brief 在笛卡尔网格上生成二维界面拟合网格

    @param mesh 结构四边形网格
    """

    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()

    node = mesh.entity('node')
    edge = mesh.entity('edge')
    cell = mesh.entity('cell')
    edge2cell = mesh.ds.edge_to_cell()

    if callable(interface):
        phi = interface(node)
    else:
        phi = interface
    sphi = msign(phi)

    isCutEdge = sphi[edge[:, 0]]*sphi[edge[:, 1]] < 0
    isCutCell = np.zeros(NC, dtype=np.bool_)
    isCutCell[edge2cell[isCutEdge, 0:2]] = True
    flag = (np.sum(np.abs(sphi[cell]), axis=1) < 3)
    isCutCell[flag] = True

    A = node[edge[isCutEdge, 0]]
    B = node[edge[isCutEdge, 1]]

    if callable(interface):
        cnode = find_cut_point(interface, A, B)
    else:
        phiA = np.abs(phi[edge[isCutEdge, 0]])
        phiB = np.abs(phi[edge[isCutEdge, 1]])
        l = phiA + phiB
        l0 = phiB/l
        l1 = phiA/l
        cnode = l0[:, None]*A + l1[:, None]*B

    node = np.r_['0', node, cnode]

    NCN = len(cnode) 
    phi = np.append(phi, np.zeros(NCN))
    sphi = np.append(sphi, np.zeros(NCN))

    isSpecialCell = (np.sum(np.abs(sphi[cell]), axis=1) == 2) & (np.sum(sphi[cell], axis=1) == 0)

    scell = cell[isSpecialCell]
    anode = node[scell].sum(axis=1)/4

    if callable(interface):
        anode = interface.project(anode)

    NAN = len(anode)
    node = np.r_['0', node, anode]

    phi = np.append(phi, np.zeros(NAN))
    sphi = np.append(sphi, np.zeros(NAN))

    isInterfaceNode = np.zeros(NN+NCN+NAN, dtype=np.bool_)

    isInterfaceNode[cell[isCutCell]] = True
    isInterfaceNode[NN:] = True

    inode = node[isInterfaceNode]
    idxMap, = np.nonzero(isInterfaceNode)
    t = Delaunay(inode)
    tcell = t.simplices

    NI = len(inode) - NCN - NAN
    isUnnecessaryCell = (np.sum(tcell < NI, axis=1) == 3)
    tcell = idxMap[tcell[~isUnnecessaryCell, :]]

    scell = cell[~isCutCell]

    cell = np.r_['0', tcell, scell[:, [1, 2, 0]], scell[:, [3, 0, 2]]]

    return TriangleMesh(node, cell)


def msign(x):
    flag = np.sign(x)
    flag[np.abs(x) < 1e-8] = 0
    return flag



def datastructure(cell):
    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    totalEdge = cell[:, localEdge].reshape(-1, 2)
    _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
            return_index=True,
            return_inverse=True,
            axis=0)
    NE = i0.shape[0]
    edge2cell = np.zeros((NE, 4), dtype=cell.dtype)

    i1 = np.zeros(NE, dtype=cell.dtype)
    i1[j] = range(3*NC)

    edge2cell[:, 0] = i0//3
    edge2cell[:, 1] = i1//3
    edge2cell[:, 2] = i0%3
    edge2cell[:, 3] = i1%3

    edge = totalEdge[i0, :]

    return edge, edge2cell
    
    


