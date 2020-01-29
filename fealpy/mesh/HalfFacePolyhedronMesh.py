import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from ..quadrature import TriangleQuadrature
from .Mesh2d import Mesh2d
from .adaptive_tools import mark


class HalfFacePolyhedronMesh():

    def __init__(self, node, halfedge, halfface, NC):
        pass

    @classmethod
    def from_mesh(self, mesh):
        NN = mesh.number_of_nodes()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        FE = mesh.ds.FE
        face2cell = mesh.ds.face_to_cell()
        halfface = np.zeros((2*NF, 4), dtype=mesh.itype)
        halfedge = np.zeros((2*NF*FE, 6), dtype=mesh.itype)

        isBdFace = (face2cell[:, 0] == face2cell[:, 1])
        halfface[:, 0] = range(0, 2*NF*FE, 3)
        halfface[:, 1] = face2cell[:, 0:2].flat
        halfface[1::2, 1][isBdFace] = NC
        halfface[0::2, 2] = range(1, 2*NF, 2)
        halfface[1::2, 2] = range(0, 2*NF, 2)
        halfface[0::2, 3] = 1

        idx = np.zeros(2*FE, dtype=mesh.itype)
        idx[:FE] = np.arange(1, FE+1)%FE
        idx[FE:] = range(FE)
        halfedge[:, 0] = face[:, idx].flat
        idx = np.arange(2*NF)
        halfedge[:, 1] = np.repeat(idx, FE)
        halfedge[:, 2] = 0
        halfedge[:, 3] = 0
        halfedge[:, 4] = 0
        halfedge[:, 5] = 0

class HalfFacePolygonMeshDataStructure():
    def __init__(self, NN, NE, NF, NC, halfedge, halfface):
        self.NN = NN
        self.NE = NE
        self.NC = NC
        self.NF = len(halfface)//2
        self.halfedge = halfedge
        self.halfface = halfface
        self.itype = halfedge.dtype

        self.cell2hface = np.zeros(NC+1, dtype=self.itype)
        self.cell2hface[halfface[:, 1]] = range(halfface.shape[0])
        self.hface2hedge[halfedge[:, 1]] = range(halfedge.shape[0])

    def reinit(self, NN, NE, NF, NC, halfedge):
        self.NN = NN
        self.NE = NE
        self.NC = NC
        self.NF = len(halfface)//2
        self.halfedge = halfedge
        self.halfface = halfface
        self.itype = halfedge.dtype

        self.cell2hface = np.zeros(NC+1, dtype=self.itype)
        self.cell2hface[halfface[:, 1]] = range(halfface.shape[0])
        self.hface2hedge[halfedge[:, 1]] = range(halfedge.shape[0])
