import numpy as np
from .adaptive_tools import mark


class HalfEdgeMesh3d():
    def __init__(self, node, halfedge, subdomain):
        """
        Parameters
        ----------
            halfedge:
                halfedge[i, 0] : index of the node pointed by i-th halfedge
                halfedge[i, 1] : index of the face enclosed by i-th halfedge
                halfedge[i, 2] : index of the cell enclosed by of i-th halfedge
                halfedge[i, 3] : index of the next halfedge of i-th halfedge
                halfedge[i, 4] : index of the prev halfedge of i-th halfedge
                halfedge[i, 5] : index of the opposite halfedge of i-th halfedge, 
                    and the halfedge[i, 5]-th and i-th halfedge are in the same cell.
                halfedge[i, 6] : index of the opposite halfedge of i-th halfedge, 
                    and the halfedge[i, 6]-th and i-th halfedge in the oposite face.
            subdomain: (NC, )the sub domain flag of each cell blong to
        """
        pass

    @classmethod
    def from_mesh(self, mesh):
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        FE = mesh.ds.FE # number of edges of each face
        face2cell = mesh.ds.face_to_cell()
        halfedge = np.zeros((2*NF*FE, 7), dtype=mesh.itype)

        isBdFace = (face2cell[:, 0] == face2cell[:, 1])
        
        idx0 = np.arange(FE)
        halfedge[0::2, 0] = face[:, idx0].flat
        halfedge[1::2, 0] = face[:, idx1].flat


class HalfEdgeMesh3dDataStructure():
    def __init__(self, NN, NE, NF, NC, halfedge, edge):
        self.reinit(NN, NE, NF, NC, halfedge, edge)

    def reinit(self, NN, NE, NF, NC, halfedge, edge):
        self.NN = NN
        self.NE = NE
        self.NC = NC
        self.NF = NF 
        self.halfedge = halfedge
        self.itype = halfedge.dtype

        self.cell = np.zeros(NC, dtype=self.itype) # cell[i] is the index of one face of i-th cell
        self.face = np.zeros(NF, dtype=self.itype) # face[i] is the index of one halfedge of i-th face 
        self.edge = edge # edge[i] is the index of one halfedge of i-th edge
