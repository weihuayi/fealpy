"""This is the docstring for the interface_mesh_generator.py module.
"""
import numpy as np

from scipy.spatial import Delaunay

from .polygon_mesh import PolygonMesh

from .uniform_mesh_3d import UniformMesh3d
from .uniform_mesh_2d import UniformMesh2d
from .tetrahedron_mesh import TetrahedronMesh
from .polyhedron_mesh import PolyhedronMesh
from .triangle_mesh import TriangleMesh


def msign(x):
    flag = np.sign(x)
    flag[np.abs(x) < 1e-8] = 0
    return flag


def find_cut_point(phi, p0, p1):
    """ Find cutted point between edge `(p0, p1)` and the curve `phi`
    
    Parameters
    ----------
    phi : function
        This is a Sign distance function.
    p0 : nd.ndarray, Nx2
        p0 is leftpotint of an edge.
    p1 : nd.ndarray, Nx2  
        p1 is rightpoint of an edge.

    Returns
    -------
    cutpoint : numpy.ndarray, Nx2 
        The return value 'cutpoint' of type is float and 'cutpoint' is a
        Intersection between edge '(p0,p1)' and the curve 'phi'
        
    Raises
    ------
        BadException
        'eps' is a very small number,This is done to prevent a
        situation equal to zero and ignore the point.

    """
    cutPoint = (p0 + p1) / 2.0
    phi0 = phi(p0)
    phi1 = phi(p1)
    phic = phi(cutPoint)

    isLeft = np.zeros(p0.shape[0], dtype=np.bool_)
    isRight = np.zeros(p0.shape[0], dtype=np.bool_)
    vec = p1 - p0
    h = np.sqrt(np.sum(vec ** 2, axis=1))

    eps = np.finfo(p0.dtype).eps
    tol = np.sqrt(eps) * h * h
    isNotOK = (h > tol) & (phic != 0)
    while np.any(isNotOK):
        cutPoint[isNotOK, :] = (p0[isNotOK, :] + p1[isNotOK, :]) / 2
        phic[isNotOK] = phi(cutPoint[isNotOK, :])
        isLeft[isNotOK] = phi0[isNotOK] * phic[isNotOK] > 0
        isRight[isNotOK] = phi1[isNotOK] * phic[isNotOK] > 0
        p0[isLeft, :] = cutPoint[isLeft, :]
        p1[isRight, :] = cutPoint[isRight, :]

        phi0[isLeft] = phic[isLeft]
        phi1[isRight] = phic[isRight]
        h[isNotOK] /= 2
        isNotOK[isNotOK] = (h[isNotOK] > tol[isNotOK]) & (phic[isNotOK] != 0)
        isLeft[:] = False
        isRight[:] = False
    return cutPoint


def interfacemesh2d(box, phi, nx, ny):
    """ Generate a interface-fitted mesh 

    Parameters
    ----------
    box : int
        This is a box with 'x0,x1,y0,y1'
    phi : function
        This is a sign distance function
    n : int
        'n' is number of segment

    Returns
    -------
    hx : float
        'hx' is the split step of the x-axis
    hy : float
        'hy' is the split step of the y-axis
    h : float
        'h' is the split step 
    """

    hx = (box[1] - box[0]) / nx
    hy = (box[3] - box[2]) / ny
    h = min(hx, hy)

    mesh = UniformMesh2d((0, nx, 0, ny), ((box[1] - box[0]) / nx, (box[3] -
        box[2]) / ny), (box[0], box[2]))

    N = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    NE = mesh.number_of_edges()

    node = mesh.entity('node')
    cell = mesh.entity('cell')[:, [0, 2, 3, 1]]

    phiValue = phi(node)
    # phiValue[np.abs(phiValue) < 0.1*h**2] = 0.0
    phiSign = np.sign(phiValue)

    # Step 1: find the nodes near interface
    edge = mesh.ds.edge
    isCutEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0
    e0 = node[edge[isCutEdge, 0]]
    e1 = node[edge[isCutEdge, 1]]
    cutNode = find_cut_point(phi, e0, e1)
    ncut = cutNode.shape[0]

    # find interface cell and point
    isInterfaceCell = np.zeros(NC, dtype=np.bool_)
    edge2cell = mesh.ds.edge_to_cell()
    isInterfaceCell[edge2cell[isCutEdge, 0:2]] = True
    isInterfaceCell[np.sum(np.abs(phiSign[cell]), axis=1) < 3] = True
    isInterfaceNode = np.zeros(N, dtype=np.bool_)
    isInterfaceNode[cell[isInterfaceCell, :]] = True

    # Find specical cells
    isSpecialCell = (np.sum(np.abs(phiSign[cell]), axis=1) == 2) \
                    & (np.sum(phiSign[cell], axis=1) == 0)
    scell = cell[isSpecialCell, :]
    auxNode = (node[scell[:, 0], :] + node[scell[:, 2], :]) / 2
    naux = auxNode.shape[0]

    interfaceNode = np.concatenate(
        (node[isInterfaceNode, :], cutNode, auxNode),
        axis=0)
    dt = Delaunay(interfaceNode)
    tri = dt.simplices
    NI = np.sum(isInterfaceNode)
    isUnnecessaryCell = (np.sum(tri < NI, axis=1) == 3)
    tri = tri[~isUnnecessaryCell, :]

    interfaceNodeIdx = np.zeros(interfaceNode.shape[0], dtype=np.int)
    interfaceNodeIdx[:NI], = np.nonzero(isInterfaceNode)
    interfaceNodeIdx[NI:NI + ncut] = N + np.arange(ncut)
    interfaceNodeIdx[NI + ncut:] = N + ncut + np.arange(naux)
    tri = interfaceNodeIdx[tri]

    # Get the final mesh in PolygonMesh 

    NS = np.sum(~isInterfaceCell)
    NT = tri.shape[0]
    pnode = np.concatenate((node, cutNode, auxNode), axis=0)
    pcell = np.zeros(NS * 4 + NT * 3, dtype=np.int)
    pcellLocation = np.zeros(NS + NT + 1, dtype=np.int)

    sview = pcell[:4 * NS].reshape(NS, 4)
    sview[:] = cell[~isInterfaceCell, :]

    tview = pcell[4 * NS:].reshape(NT, 3)
    tview[:] = tri
    pcellLocation[:NS] = range(0, 4 * NS, 4)
    pcellLocation[NS:-1] = range(4 * NS, 4 * NS + 3 * NT, 3)
    pcellLocation[-1] = 4 * NS + 3 * NT
    pmesh = PolygonMesh(pnode, pcell, pcellLocation)

    pmesh.nodeMarker = np.zeros(pmesh.number_of_nodes(), dtype=np.int)
    pmesh.nodeMarker[:N] = phiSign

    pmesh.cellMarker = np.ones(pmesh.number_of_cells(), dtype=np.int)
    pmesh.cellMarker[phi(pmesh.entity_barycenter()) < 0] = -1

    return pmesh


class InterfaceMesh2d():

    def __init__(self, interface, box=[0, 1, 0, 1], n=10):
        self.h = (box[1] - box[0]) / n

        self.interface = interface
        self.mesh = UniformMesh2d((0, n, 0, n), ((box[1] - box[0]) / n, (box[3] - box[2]) / n), (box[0], box[2]))
        self.node = self.mesh.entity('node')
        self.phi = interface(self.node)
        self.phi[np.abs(self.phi) < 0.1 * self.h ** 2] = 0.0
        self.phiSign = msign(self.phi)

        self.box = box
        self.n = n
        self.N0 = self.node.shape[0]

    def find_cut_cell(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell = mesh.ds.cell
        edge = mesh.ds.edge
        cell2edge = mesh.ds.cell_to_edge()

        self.mesh.cellFlag = np.zeros(NC, dtype=np.int)

        phiSign = self.phiSign

        isCutEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0
        isCutCell = np.zeros(NC, dtype=np.bool_)

        edge2cell = mesh.ds.edge_to_cell()
        isCutCell[edge2cell[isCutEdge, 0:2]] = True
        isCutCell[np.sum(np.abs(phiSign[cell]), axis=1) < 3] = True

        self.mesh.cellFlag[(~isCutCell) & (np.max(phiSign[cell], axis=1) == 1)] = 1
        self.mesh.cellFlag[(~isCutCell) & (np.min(phiSign[cell], axis=1) == -1)] = -1

    def is_cut_cell(self):
        cellFlag = self.mesh.cellFlag
        isCutCell = (cellFlag == 0)
        return isCutCell

    def find_cut_point(self):
        mesh = self.mesh
        node = self.node
        edge = mesh.ds.edge
        phiSign = self.phiSign
        isCutEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0

        A = node[edge[isCutEdge, 0]]
        B = node[edge[isCutEdge, 1]]

        self.N1 = self.N0 + A.shape[0]

        cutNode = find_cut_point(self.interface, A, B)
        self.node = np.append(node, cutNode, axis=0)
        self.phi = np.append(self.phi, np.zeros(A.shape[0]))
        self.phiSign = np.append(self.phiSign, np.zeros(A.shape[0]))

    def find_aux_point(self):
        mesh = self.mesh
        node = self.node
        cell = mesh.ds.cell
        phiSign = self.phiSign

        isSpecialCell = (np.sum(np.abs(phiSign[cell]), axis=1) == 2) \
                        & (np.sum(phiSign[cell], axis=1) == 0)
        scell = cell[isSpecialCell, :]

        auxNode = (node[scell[:, 0], :] + node[scell[:, 2], :]) / 2

        self.N2 = self.N1 + auxNode.shape[0]
        self.node = np.append(node, auxNode, axis=0)

        self.phi = np.append(self.phi, np.zeros(self.N2 - self.N1))
        self.phiSign = np.append(self.phiSign, np.zeros(self.N2 - self.N1))

    def find_interface_node(self):
        N = self.N2
        cell = self.mesh.ds.cell
        node = self.node
        isInterfaceNode = np.zeros(N, dtype=np.bool_)
        isCutCell = self.is_cut_cell()
        isInterfaceNode[cell[isCutCell]] = True
        isInterfaceNode[self.N0:] = True

        return node[isInterfaceNode], np.nonzero(isInterfaceNode)[0]

    def delaunay(self, interfaceNode, idxMap):
        t = Delaunay(interfaceNode)
        tcell = t.simplices
        NI = interfaceNode.shape[0] - (self.N2 - self.N0)

        isUnnecessaryCell = (np.sum(tcell < NI, axis=1) == 3)
        tcell = idxMap[tcell[~isUnnecessaryCell, :]]

        isCutCell = self.is_cut_cell()
        NS = np.sum(~isCutCell)
        NT = tcell.shape[0]
        pnode = self.node
        pcell = np.zeros(NS * 4 + NT * 3, dtype=np.int)
        pcellLocation = np.zeros(NS + NT + 1, dtype=np.int)

        cell = self.mesh.ds.cell[:, [0, 2, 3, 1]]
        sview = pcell[:4 * NS].reshape(NS, 4)
        sview[:] = cell[~isCutCell, :]

        tview = pcell[4 * NS:].reshape(NT, 3)
        tview[:] = tcell
        pcellLocation[:NS] = range(0, 4 * NS, 4)
        pcellLocation[NS:-1] = range(4 * NS, 4 * NS + 3 * NT, 3)
        pcellLocation[-1] = 4 * NS + 3 * NT

        return pnode, pcell, pcellLocation

    def run(self):
        self.find_cut_cell()

        self.find_cut_point()

        self.find_aux_point()

        interfaceNode, idxMap = self.find_interface_node()

        pnode, pcell, pcellLocation = self.delaunay(interfaceNode, idxMap)

        return pnode, pcell, pcellLocation

    def node_marker(self):
        return self.phiSign == 0


class InterfaceMesh3d():

    def __init__(self, interface, box, n):
        self.interface = interface
        self.mesh = UniformMesh3d((0, n, 0, n, 0, n),
                                  ((box[1] - box[0]) / n, (box[3] - box[2]) / n, (box[5] - box[4]) / n),
                                  (box[0], box[2], box[4]))
        self.node = self.mesh.node
        self.phi = interface(self.node)
        self.phiSign = msign(self.phi)

        self.box = box
        self.n = n

        self.h = (box[1] - box[0]) / n

        self.N0 = self.mesh.number_of_nodes()

    def get_cell_idx(self, p):
        box = self.box
        n = self.n

        x = p[:, 0] - box[0]
        y = p[:, 1] - box[2]
        z = p[:, 2] - box[4]

        h = self.h

        i = np.floor(x / h).astype(np.int)
        j = np.floor(y / h).astype(np.int)
        k = np.floor(z / h).astype(np.int)

        cellIdx = i * n * n + j * n + k

        return cellIdx.astype(np.int)

    def find_cut_cell(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell = mesh.ds.cell
        edge = mesh.ds.edge
        cell2edge = mesh.ds.cell_to_edge()

        self.cellFlag = np.zeros(NC, dtype=np.int)

        phiSign = self.phiSign

        isInterfaceEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] <= 0
        isCutCell = (np.sum(isInterfaceEdge[cell2edge], axis=1) > 0)

        self.cellFlag[(~isCutCell) & (np.max(phiSign[cell], axis=1) == 1)] = 1
        self.cellFlag[(~isCutCell) & (np.min(phiSign[cell], axis=1) == -1)] = -1

    def find_cut_node(self):
        mesh = self.mesh
        node = self.node
        edge = mesh.ds.edge
        phiSign = self.phiSign
        isCutEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0

        A = node[edge[isCutEdge, 0]]
        B = node[edge[isCutEdge, 1]]

        self.N1 = self.N0 + A.shape[0]

        cutNode = find_cut_point(self.interface, A, B)
        self.node = np.append(node, cutNode, axis=0)
        self.phi = np.append(self.phi, np.zeros(A.shape[0]))
        self.phiSign = np.append(self.phiSign, np.zeros(A.shape[0]))

    def find_aux_node(self):
        mesh = self.mesh
        node = self.node
        face = mesh.ds.face
        face2cell = mesh.ds.face_to_cell()
        phiSign = self.phiSign
        isAuxFace = (phiSign[face[:, 0]] == 0) & (phiSign[face[:, 2]] == 0)
        isAuxFace = isAuxFace | ((phiSign[face[:, 1]] == 0) & (phiSign[face[:, 3]] == 0))
        auxNode = (node[face[isAuxFace, 0]] + node[face[isAuxFace, 2]]) / 2

        self.N2 = self.N1 + auxNode.shape[0]
        self.node = np.r_['0', node, auxNode]

        self.phi = np.r_[self.phi, np.zeros(self.N2 - self.N1)]
        self.phiSign = np.r_[self.phiSign, np.zeros(self.N2 - self.N1)]

    def is_cut_cell(self):
        cellFlag = self.cellFlag
        isCutCell = (cellFlag == 0)
        return isCutCell

    def find_interface_node(self):
        N = self.N2
        cell = self.mesh.ds.cell
        node = self.node
        isInterfaceNode = np.zeros(N, dtype=np.bool_)
        isCutCell = self.is_cut_cell()
        isInterfaceNode[cell[isCutCell]] = True
        isInterfaceNode[self.N0:] = True

        return node[isInterfaceNode], np.nonzero(isInterfaceNode)[0]

    def delaunay(self, interfaceNode, idxMap):

        t = Delaunay(interfaceNode)
        tcell = t.simplices

        bc = np.sum(interfaceNode[tcell], axis=1).reshape(-1, 3) / 4

        cellIdx = self.get_cell_idx(bc)

        # Get rid of tets on the cube boundary
        eps = 1e-12
        box = self.box
        isBadCell = (bc[:, 0] > box[1] - eps) | (bc[:, 0] < box[0] + eps)
        isBadCell = isBadCell | (bc[:, 1] > box[3] - eps) | (bc[:, 1] < box[2] + eps)
        isBadCell = isBadCell | (bc[:, 2] > box[5] - eps) | (bc[:, 2] < box[4] + eps)

        tcell = tcell[~isBadCell]
        bc = bc[~isBadCell]

        # Get rid of the tets out of the cutted cubes

        isCutCell = self.is_cut_cell()

        tcell = tcell[isCutCell[cellIdx]]
        cellIdx = cellIdx[isCutCell[cellIdx]]

        h = self.h
        X = interfaceNode[tcell, 0]
        Y = interfaceNode[tcell, 1]
        Z = interfaceNode[tcell, 2]
        isBadCell = (np.max(X, axis=1) - np.min(X, axis=1)) > 2 * h - eps
        isBadCell = isBadCell | ((np.max(Y, axis=1) - np.min(Y, axis=1)) > 2 * h - eps)
        isBadCell = isBadCell | ((np.max(Z, axis=1) - np.min(Z, axis=1)) > 2 * h - eps)

        tcell = tcell[~isBadCell]
        cellIdx = cellIdx[~isBadCell]

        T = TetrahedronMesh(self.node, idxMap[tcell])
        T.cellIdx = cellIdx

        return T

    def tet_to_poly(self, T):
        # Get the interface and triangles between cubes 
        cellIdx = T.cellIdx

        tcell = T.ds.cell
        face = T.ds.face
        face2cell = T.ds.face_to_cell()

        phiSign = self.phiSign
        isInteriorTet = np.min(phiSign[tcell], axis=1) == -1

        isInterfaceFace0 = (face2cell[:, 0] != face2cell[:, 1]) & isInteriorTet[face2cell[:, 0]] & (
            ~isInteriorTet[face2cell[:, 1]])
        isInterfaceFace1 = (face2cell[:, 0] != face2cell[:, 1]) & (~isInteriorTet[face2cell[:, 0]]) & isInteriorTet[
            face2cell[:, 1]]

        tface0 = face[isInterfaceFace0]
        tface0 = np.append(tface0, face[isInterfaceFace1][:, [0, 2, 1]], axis=0)

        n = np.sum(isInterfaceFace0)
        tidx0 = np.zeros((tface0.shape[0], 2), dtype=np.int)
        tidx0[:n, 0] = cellIdx[face2cell[isInterfaceFace0, 0]]
        tidx0[:n, 1] = cellIdx[face2cell[isInterfaceFace0, 1]]
        tidx0[n:, 0] = cellIdx[face2cell[isInterfaceFace1, 1]]
        tidx0[n:, 1] = cellIdx[face2cell[isInterfaceFace1, 0]]

        isTriFace1 = (cellIdx[face2cell[:, 0]] != cellIdx[face2cell[:, 1]]) & (~isInterfaceFace0) & (~isInterfaceFace1)
        tface1 = face[isTriFace1]
        tidx1 = cellIdx[face2cell[isTriFace1, 0:2]]

        # Get the boundary square faces of cutted cubes
        mesh = self.mesh
        face = mesh.ds.face
        face2cell = mesh.ds.face_to_cell()
        isCutCell = self.is_cut_cell()

        isSquareFace0 = isCutCell[face2cell[:, 0]] & ((~isCutCell[face2cell[:,
                                                                  1]]) | (face2cell[:, 0] == face2cell[:, 1]))
        isSquareFace1 = isCutCell[face2cell[:, 1]] & ((~isCutCell[face2cell[:,
                                                                  0]]) | (face2cell[:, 0] == face2cell[:, 1]))

        sface0 = face[isSquareFace0]
        sidx0 = face2cell[isSquareFace0, 0]

        sface1 = face[isSquareFace1]
        sface1 = sface1[:, [0, 3, 2, 1]]
        sidx1 = face2cell[isSquareFace1, 1]

        # re-numbering 
        NC = mesh.number_of_cells()
        tidx0[:, 1] += NC

        phiSign = self.phiSign

        isExTFace = (np.min(phiSign[tface1], axis=1) >= 0)
        tidx1[isExTFace] += NC

        isExSFace = (np.min(phiSign[sface0], axis=1) >= 0)
        sidx0[isExSFace] += NC

        isExSFace = (np.min(phiSign[sface1], axis=1) >= 0)
        sidx1[isExSFace] += NC

        sface = np.concatenate((sface0, sface1), axis=0)
        sidx = np.concatenate((sidx0, sidx1), axis=0)

        # 
        NF0 = tface0.shape[0]
        NF1 = tface1.shape[0]
        NF2 = sface.shape[0]

        PNF = NF0 + NF1 + NF2

        NVS = 3 * (NF0 + NF1) + 4 * NF2
        pfaceLocation = np.zeros(PNF + 1, dtype=np.int)
        pfaceLocation[0:(NF0 + NF1)] = range(0, 3 * (NF0 + NF1), 3)
        pfaceLocation[(NF0 + NF1):] = range(3 * (NF0 + NF1), NVS + 1, 4)

        pface = np.zeros(NVS, dtype=np.int)

        pface[0:3 * NF0] = tface0.flatten()
        pface[3 * NF0:3 * (NF0 + NF1)] = tface1.flatten()
        pface[3 * (NF0 + NF1):NVS] = sface.flatten()

        pface2cell = np.zeros((PNF, 2), dtype=np.int)
        pface2cell[0:NF0, :] = tidx0
        pface2cell[NF0:NF0 + NF1, :] = tidx1
        pface2cell[NF0 + NF1:, :] = sidx.reshape(-1, 1)

        flag = np.ones(PNF, dtype=np.int)
        flag[NF0:NF0 + NF1] = 2
        flag[NF0 + NF1:] = 3

        NP = np.max(pface2cell) + 1

        isCutPoly = np.zeros(NP, dtype=np.bool_)
        isCutPoly[pface2cell] = True

        cutPolyIdx = np.zeros(NP, dtype=np.int)
        NP = isCutPoly.sum()
        cutPolyIdx[isCutPoly] = range(NP)

        pface2cell = cutPolyIdx[pface2cell]
        pmesh = PolyhedronMesh(self.node, pface, pfaceLocation, pface2cell)

        pmesh.cellData = {'flag': flag, }

        return pmesh

    def interface_mesh(self, T):
        cellIdx = T.cellIdx

        tcell = T.ds.cell
        face = T.ds.face
        face2cell = T.ds.face_to_cell()

        phiSign = self.phiSign
        isInteriorTet = np.min(phiSign[tcell], axis=1) == -1

        isInterfaceFace0 = (face2cell[:, 0] != face2cell[:, 1]) & isInteriorTet[face2cell[:, 0]] & (
            ~isInteriorTet[face2cell[:, 1]])
        isInterfaceFace1 = (face2cell[:, 0] != face2cell[:, 1]) & (~isInteriorTet[face2cell[:, 0]]) & isInteriorTet[
            face2cell[:, 1]]

        tface0 = face[isInterfaceFace0]
        tface0 = np.append(tface0, face[isInterfaceFace1][:, [0, 2, 1]], axis=0)

        node = T.node
        N = T.number_of_nodes()
        isInterfaceNode = np.zeros(N, dtype=np.bool_)
        isInterfaceNode[tface0] = True
        NN = np.sum(isInterfaceNode)
        idxMap = np.zeros(N, dtype=np.int)
        idxMap[isInterfaceNode] = range(NN)
        triangles = idxMap[tface0]

        return TriangleMesh(node[isInterfaceNode], triangles)

    def run(self, meshtype='polyhedron'):

        self.find_cut_cell()

        self.find_cut_node()

        self.find_aux_node()

        interfaceNode, idxMap = self.find_interface_node()

        T = self.delaunay(interfaceNode, idxMap)

        if meshtype == 'polyhedron':
            mesh = self.tet_to_poly(T)
        elif meshtype == 'interfacemesh':
            mesh = self.interface_mesh(T)
        return mesh
