"""This is the docstring for the interface_mesh_generator.py module.
"""
import numpy as np

from scipy.spatial import Delaunay


from .PolygonMesh import PolygonMesh
from .PolyhedronMesh import PolyhedronMesh 


from .StructureHexMesh import StructureHexMesh
from .StructureQuadMesh import StructureQuadMesh 
from .TetrahedronMesh import TetrahedronMesh
from .PolyhedronMesh import PolyhedronMesh 


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
    cutPoint = (p0+p1)/2.0
    phi0 = phi(p0)
    phi1 = phi(p1)
    phic = phi(cutPoint)

    isLeft = np.zeros(p0.shape[0], dtype=np.bool)
    isRight = np.zeros(p0.shape[0], dtype=np.bool)
    vec = p1 - p0
    h = np.sqrt(np.sum(vec**2, axis=1))

    eps = np.finfo(p0.dtype).eps
    tol = np.sqrt(eps)*h*h
    isNotOK = (h > tol) & (phic != 0)
    while np.any(isNotOK):
        cutPoint[isNotOK, :] = (p0[isNotOK, :] + p1[isNotOK,:])/2
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

def interfacemesh2d(box, phi, n):
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

    hx = (box[1] - box[0])/n
    hy = (box[3] - box[2])/n
    h = min(hx, hy)

    mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad') 

    N = mesh.number_of_points()
    NC = mesh.number_of_cells()
    NE = mesh.number_of_edges()

    point = mesh.point
    cell = mesh.ds.cell

    phiValue = phi(point)
    #phiValue[np.abs(phiValue) < 0.1*h**2] = 0.0
    phiSign = np.sign(phiValue)

    # Step 1: find the points near interface
    edge = mesh.ds.edge
    isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0 
    e0 = point[edge[isCutEdge, 0]]
    e1 = point[edge[isCutEdge, 1]]
    cutPoint = find_cut_point(phi, e0, e1)
    ncut = cutPoint.shape[0]

    # find interface cell and point
    isInterfaceCell = np.zeros(NC, dtype=np.bool)
    edge2cell =  mesh.ds.edge_to_cell()
    isInterfaceCell[edge2cell[isCutEdge, 0:2]] = True
    isInterfaceCell[np.sum(np.abs(phiSign[cell]), axis=1) < 3] = True
    isInterfacePoint = np.zeros(N, dtype=np.bool)
    isInterfacePoint[cell[isInterfaceCell,:]] = True

    # Find specical cells
    isSpecialCell = (np.sum(np.abs(phiSign[cell]), axis=1) == 2) \
            & (np.sum(phiSign[cell], axis=1) == 0)
    scell = cell[isSpecialCell, :]
    auxPoint = (point[scell[:, 0], :] + point[scell[:, 2], :])/2
    naux = auxPoint.shape[0]

    interfacePoint = np.concatenate(
            (point[isInterfacePoint, :], cutPoint, auxPoint), 
            axis=0)
    dt = Delaunay(interfacePoint)
    tri = dt.simplices 
    NI = np.sum(isInterfacePoint)
    isUnnecessaryCell = (np.sum(tri < NI, axis=1) == 3)
    tri = tri[~isUnnecessaryCell, :]

    interfacePointIdx = np.zeros(interfacePoint.shape[0], dtype=np.int)
    interfacePointIdx[:NI], = np.nonzero(isInterfacePoint) 
    interfacePointIdx[NI:NI+ncut] = N + np.arange(ncut)
    interfacePointIdx[NI+ncut:] = N + ncut + np.arange(naux)
    tri = interfacePointIdx[tri]

    # Get the final mesh in PolygonMesh 

    NS = np.sum(~isInterfaceCell)
    NT = tri.shape[0]
    ppoint = np.concatenate((point, cutPoint, auxPoint), axis=0)
    pcell = np.zeros(NS*4 + NT*3, dtype=np.int) 
    pcellLocation = np.zeros(NS + NT + 1, dtype=np.int) 

    sview = pcell[:4*NS].reshape(NS, 4)
    sview[:] = cell[~isInterfaceCell,:]

    tview = pcell[4*NS:].reshape(NT, 3)
    tview[:] = tri
    pcellLocation[:NS] = range(0, 4*NS, 4)
    pcellLocation[NS:-1] = range(4*NS, 4*NS+3*NT, 3)
    pcellLocation[-1] = 4*NS+3*NT
    pmesh = PolygonMesh(ppoint, pcell, pcellLocation)

    pmesh.pointMarker = np.zeros(pmesh.number_of_points(), dtype=np.int)
    pmesh.pointMarker[:N] = phiSign

    pmesh.cellMarker = np.ones(pmesh.number_of_cells(), dtype=np.int)
    pmesh.cellMarker[phi(pmesh.barycenter()) < 0] = -1

    return pmesh    

    
class InterfaceMesh2d():

    def __init__(self, interface, box, n):
        self.h = (box[1] - box[0])/n

        self.interface = interface 
        self.mesh = StructureQuadMesh(box, nx=n, ny=n)
        self.point = self.mesh.point
        self.phi = interface(self.point)
        self.phi[np.abs(self.phi) < 0.1*self.h**2] = 0.0
        self.phiSign = msign(self.phi) 

        self.box = box
        self.n = n
        self.N0 = self.point.shape[0]

    def find_cut_cell(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell = mesh.ds.cell
        edge = mesh.ds.edge
        cell2edge = mesh.ds.cell_to_edge()

        self.mesh.cellFlag = np.zeros(NC, dtype=np.int)

        phiSign = self.phiSign 

        isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
        isCutCell = np.zeros(NC, dtype=np.bool)

        edge2cell =  mesh.ds.edge_to_cell()
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
        point = self.point
        edge = mesh.ds.edge
        phiSign = self.phiSign 
        isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
        
        A = point[edge[isCutEdge, 0]]
        B = point[edge[isCutEdge, 1]]

        self.N1 = self.N0 + A.shape[0]

        cutPoint = find_cut_point(self.interface, A, B)
        self.point = np.append(point, cutPoint, axis=0)
        self.phi = np.append(self.phi, np.zeros(A.shape[0]))
        self.phiSign = np.append(self.phiSign, np.zeros(A.shape[0]))

    def find_aux_point(self):
        mesh = self.mesh
        point = self.point
        cell = mesh.ds.cell
        phiSign = self.phiSign 

        isSpecialCell = (np.sum(np.abs(phiSign[cell]), axis=1) == 2) \
                & (np.sum(phiSign[cell], axis=1) == 0)
        scell = cell[isSpecialCell, :]

        auxPoint = (point[scell[:, 0], :] + point[scell[:, 2], :])/2

        self.N2 = self.N1 + auxPoint.shape[0]
        self.point = np.append(point, auxPoint, axis=0)

        self.phi = np.append(self.phi, np.zeros(self.N2 - self.N1))
        self.phiSign = np.append(self.phiSign, np.zeros(self.N2 - self.N1))

    def find_interface_point(self):
        N = self.N2 
        cell = self.mesh.ds.cell
        point = self.point
        isInterfacePoint = np.zeros(N, dtype=np.bool)
        isCutCell = self.is_cut_cell()
        isInterfacePoint[cell[isCutCell]] = True
        isInterfacePoint[self.N0:] = True

        return point[isInterfacePoint], np.nonzero(isInterfacePoint)[0]

    def delaunay(self, interfacePoint, idxMap):

        t = Delaunay(interfacePoint)
        tcell = t.simplices
        NI = interfacePoint.shape[0] - (self.N2 - self.N0)

        isUnnecessaryCell = (np.sum(tcell < NI, axis=1) == 3)
        tcell = idxMap[tcell[~isUnnecessaryCell, :]]

        isCutCell = self.is_cut_cell()
        NS = np.sum(~isCutCell)
        NT = tcell.shape[0]
        ppoint = self.point 
        pcell = np.zeros(NS*4 + NT*3, dtype=np.int) 
        pcellLocation = np.zeros(NS + NT + 1, dtype=np.int) 

        cell = self.mesh.ds.cell
        sview = pcell[:4*NS].reshape(NS, 4)
        sview[:] = cell[~isCutCell,:]

        tview = pcell[4*NS:].reshape(NT, 3)
        tview[:] = tcell 
        pcellLocation[:NS] = range(0, 4*NS, 4)
        pcellLocation[NS:-1] = range(4*NS, 4*NS+3*NT, 3)
        pcellLocation[-1] = 4*NS+3*NT

        return ppoint, pcell, pcellLocation 

    def run(self):

        self.find_cut_cell()

        self.find_cut_point()

        self.find_aux_point()

        interfacePoint, idxMap = self.find_interface_point()

        ppoint, pcell, pcellLocation = self.delaunay(interfacePoint, idxMap)

        return ppoint, pcell, pcellLocation 

    def point_marker(self):
        return self.phiSign == 0 



class InterfaceMesh3d():

    def __init__(self, interface, box, n):
        self.interface = interface
        self.mesh = StructureHexMesh(box, nx=n, ny=n, nz=n)
        self.point = self.mesh.point
        self.phi = interface(self.point)
        self.phiSign = msign(self.phi)

        self.box = box
        self.n = n

        self.h = (box[1] - box[0])/n

        self.N0 = self.mesh.number_of_points()

    def get_cell_idx(self, p):
        box = self.box
        n = self.n

        x = p[:, 0] - box[0]
        y = p[:, 1] - box[2]
        z = p[:, 2] - box[4]

        h = self.h 

        i = np.floor(x/h).astype(np.int)
        j = np.floor(y/h).astype(np.int)
        k = np.floor(z/h).astype(np.int)

        cellIdx = i*n*n + j*n + k

        return cellIdx.astype(np.int)

    def find_cut_cell(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell = mesh.ds.cell
        edge = mesh.ds.edge
        cell2edge = mesh.ds.cell_to_edge()

        self.cellFlag = np.zeros(NC, dtype=np.int)

        phiSign = self.phiSign 

        isInterfaceEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] <= 0
        isCutCell = (np.sum(isInterfaceEdge[cell2edge], axis=1) > 0)

        self.cellFlag[(~isCutCell) & (np.max(phiSign[cell], axis=1) == 1)] = 1
        self.cellFlag[(~isCutCell) & (np.min(phiSign[cell], axis=1) == -1)] = -1

    def find_cut_point(self):
        mesh = self.mesh
        point = self.point
        edge = mesh.ds.edge
        phiSign = self.phiSign 
        isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
        
        A = point[edge[isCutEdge, 0]]
        B = point[edge[isCutEdge, 1]]

        self.N1 = self.N0 + A.shape[0]

        cutPoint = find_cut_point(self.interface, A, B)
        self.point = np.append(point, cutPoint, axis=0)
        self.phi = np.append(self.phi, np.zeros(A.shape[0]))
        self.phiSign = np.append(self.phiSign, np.zeros(A.shape[0]))

    def find_aux_point(self):
        mesh = self.mesh
        point = self.point
        face = mesh.ds.face
        face2cell = mesh.ds.face_to_cell()
        phiSign = self.phiSign 
        isAuxFace = (phiSign[face[:, 0]] == 0) & (phiSign[face[:, 2]] ==0)
        isAuxFace = isAuxFace | ((phiSign[face[:, 1]] == 0) & (phiSign[face[:, 3]] ==0))
        auxPoint = (point[face[isAuxFace, 0]] + point[face[isAuxFace, 2]])/2

        self.N2 = self.N1 + auxPoint.shape[0]
        self.point = np.append(point, auxPoint, axis=0)

        self.phi = np.append(self.phi, np.zeros(self.N2 - self.N1))
        self.phiSign = np.append(self.phiSign, np.zeros(self.N2 - self.N1))

    def is_cut_cell(self):
        cellFlag = self.cellFlag
        isCutCell = (cellFlag == 0)
        return isCutCell

    def find_interface_point(self):
        N = self.N2 
        cell = self.mesh.ds.cell
        point = self.point
        isInterfacePoint = np.zeros(N, dtype=np.bool)
        isCutCell = self.is_cut_cell()
        isInterfacePoint[cell[isCutCell]] = True
        isInterfacePoint[self.N0:] = True

        return point[isInterfacePoint], np.nonzero(isInterfacePoint)[0]


    def delaunay(self, interfacePoint, idxMap):

        t = Delaunay(interfacePoint)
        tcell = t.simplices

        bc = np.sum(interfacePoint[tcell], axis=1).reshape(-1, 3)/4

        cellIdx = self.get_cell_idx(bc) 

#        NI = interfacePoint.shape[0] - (self.N2 - self.N0)
#
#        isUnnecessaryCell = (np.sum(tcell < NI, axis=1) == 4)
#        tcell = tcell[~isUnnecessaryCell, :]

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
        X = interfacePoint[tcell, 0]
        Y = interfacePoint[tcell, 1]
        Z = interfacePoint[tcell, 2]
        isBadCell = (np.max(X, axis=1) - np.min(X, axis=1)) > 2*h - eps
        isBadCell = isBadCell | ((np.max(Y, axis=1) - np.min(Y, axis=1)) > 2*h - eps)
        isBadCell = isBadCell | ((np.max(Z, axis=1) - np.min(Z, axis=1)) > 2*h - eps)

        tcell = tcell[~isBadCell]
        cellIdx = cellIdx[~isBadCell]

        T = TetrahedronMesh(self.point, idxMap[tcell])
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

        isInterfaceFace0 = (face2cell[:, 0] != face2cell[:, 1]) & isInteriorTet[face2cell[:, 0]] & (~isInteriorTet[face2cell[:, 1]])   
        isInterfaceFace1 = (face2cell[:, 0] != face2cell[:, 1]) & (~isInteriorTet[face2cell[:, 0]]) & isInteriorTet[face2cell[:, 1]]

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

        isExTFace = (np.min(phiSign[tface1], axis=1) >=0)
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

        PNF= NF0 + NF1 + NF2

        NVS = 3*(NF0+NF1)+4*NF2
        pfaceLocation = np.zeros(PNF+1, dtype=np.int)
        pfaceLocation[0:(NF0+NF1)] = range(0, 3*(NF0+NF1), 3) 
        pfaceLocation[(NF0+NF1):] = range(3*(NF0+NF1), NVS+1, 4)

        pface = np.zeros(NVS, dtype=np.int)

        pface[0:3*NF0] = tface0.flatten()
        pface[3*NF0:3*(NF0+NF1)] = tface1.flatten()
        pface[3*(NF0+NF1):NVS] = sface.flatten()

        pface2cell = np.zeros((PNF, 2), dtype=np.int)
        pface2cell[0:NF0, :] = tidx0
        pface2cell[NF0:NF0+NF1, :] = tidx1
        pface2cell[NF0+NF1:, :] = sidx.reshape(-1, 1)

        flag = np.ones(PNF, dtype=np.int)
        flag[NF0:NF0+NF1] = 2 
        flag[NF0+NF1:] = 3

        NP = np.max(pface2cell)+1

        isCutPoly = np.zeros(NP, dtype=np.bool)
        isCutPoly[pface2cell] = True

        cutPolyIdx = np.zeros(NP, dtype=np.int)
        NP = isCutPoly.sum()
        cutPolyIdx[isCutPoly] = range(NP)

        pface2cell = cutPolyIdx[pface2cell]
        pmesh = PolyhedronMesh(self.point, pface, pfaceLocation, pface2cell)

        pmesh.cellData = {'flag':flag, }

        return pmesh



    def run(self):

        self.find_cut_cell()

        self.find_cut_point()

        self.find_aux_point()

        interfacePoint, idxMap = self.find_interface_point()

        T = self.delaunay(interfacePoint, idxMap)

        pmesh = self.tet_to_poly(T)

        return pmesh

