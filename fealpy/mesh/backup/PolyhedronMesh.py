import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse import triu, tril, find, hstack


class PolyhedronMesh():
    def __init__(self, node, face, faceLocation, face2cell, NC=None,
            itype=np.int_,
            ftype=np.float64):
        self.node = node
        self.ds = PolyhedronMeshDataStructure(node.shape[0], face, faceLocation, face2cell, NC=NC)
        self.meshtype = 'polyhedron'
        self.itype = itype
        self.ftype = ftype 

    def to_vtk(self):
        NF = self.number_of_faces()
        face = self.ds.face
        face2cell = self.ds.face2cell
        faceLocation = self.ds.faceLocation
        NV = self.ds.number_of_vertices_of_faces()

        faces = np.zeros(len(face) + NF, dtype=self.itype)
        isIdx = np.ones(len(face) + NF, dtype=np.bool_)
        isIdx[0] = False
        isIdx[np.add.accumulate(NV+1)[:-1]] = False
        faces[~isIdx] = NV
        faces[isIdx] = face
        return NF, faces

    def check(self):
        N = self.number_of_nodes()
        NC = self.number_of_cells()
        NFE = self.ds.number_of_edges_of_faces()
        
        face2cell = self.ds.face2cell
        isIntFace = (face2cell[:, 0] != face2cell[:, 1])

        cell2node = self.ds.cell_to_node()
        V = cell2node@np.ones(N, dtype=self.itype)
        E = np.zeros(NC, dtype=self.itype)
        F = np.zeros(NC, dtype=self.itype)

        np.add.at(E, face2cell[:, 0], NFE)
        np.add.at(E, face2cell[isIntFace, 1], NFE[isIntFace])
        E = E//2

        np.add.at(F, face2cell[:, 0], 1)
        np.add.at(F, face2cell[isIntFace, 1], 1)

        val = F - E + V
        isBadPoly = (val != 2)
        return np.any(isBadPoly)

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NF

    def number_of_cells(self):
        return self.ds.NC

    def geo_dimension(self):
        return self.node.shape[1]

    def face_angle(self):
        node = self.node
        face = self.ds.face
        faceLocation = self.ds.faceLocation

        idx1 = np.zeros(face.shape[0], dtype=self.itype)
        idx2 = np.zeros(face.shape[0], dtype=self.itype)

        idx1[0:-1] = face[1:]
        idx1[faceLocation[1:]-1] = face[faceLocation[:-1]]
        idx2[1:] = face[0:-1]
        idx2[faceLocation[:-1]] = face[faceLocation[1:]-1]
        a = node[idx1] - node[face]
        b = node[idx2] - node[face]
        la = np.sum(a**2, axis=1)
        lb = np.sum(b**2, axis=1)
        x = np.arccos(np.sum(a*b, axis=1)/np.sqrt(la*lb))
        return np.degrees(x)

    def volume(self):
        pass

    def face_area(self):
        pass

    def face_unit_normal(self):
        pass

    def edge_unit_tagent(self):
        pass

class PolyhedronMeshDataStructure():
    def __init__(self, N, face, faceLocation, face2cell, NC=None):
        self.N = N 
        self.NF = faceLocation.shape[0] - 1
        if NC is None:
            self.NC = np.max(face2cell) + 1
        else:
            self.NC = NC

        self.face = face
        self.faceLocation = faceLocation
        self.face2cell = face2cell

        self.construct()

    def reinit(self, N, face, faceLocation, face2cell, NC=None):
        self.N = N 
        self.NF = faceLocation.shape[0] - 1
        if NC is None:
            self.NC = np.max(face2cell) + 1
        else:
            self.NC = NC

        self.face = face
        self.faceLocation = faceLocation
        self.face2cell
        self.construct()

    def clear(self):
        self.edge = None 
        self.face2edge = None

    def number_of_vertices_of_faces(self):
        faceLocation = self.faceLocation 
        return faceLocation[1:] - faceLocation[0:-1] 

    def number_of_edges_of_faces(self):
        faceLocation = self.faceLocation 
        return faceLocation[1:] - faceLocation[0:-1] 

    def total_edge(self):
        face = self.face
        faceLocation = self.faceLocation

        totalEdge = np.zeros((len(face), 2), dtype=self.itype)
        totalEdge[:, 0] = face 
        totalEdge[:-1, 1] = face[1:] 
        totalEdge[faceLocation[1:] - 1, 1] = face[faceLocation[:-1]]

        return totalEdge

    def construct(self):

        totalEdge = self.total_edge()
        _, i0, j = np.unique(
                np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)

        self.NE = len(i0) 

        self.edge = totalEdge[i0]
        self.face2edge = j

        return 

    def cell_to_node(self):
        N = self.N
        NF = self.NF
        NC = self.NC

        face = self.face
        face2cell = self.face2cell

        NFV = self.number_of_vertices_of_faces()

        I = np.repeat(face2cell[:, 0], NFV)
        val = np.ones(len(face), dtype=np.bool_)
        cell2node = coo_matrix((val, (I, face)), shape=(NC, N), dtype=np.bool_)

        I = np.repeat(face2cell[:, 1], NFV)
        cell2node+= coo_matrix((val, (I, face)), shape=(NC, N), dtype=np.bool_)

        return cell2node.tocsr()

    def cell_to_edge(self):
        NC = self.NC
        NE = self.NE
        NF = self.NF 

        face = self.face
        face2edge = self.face2edge
        face2cell = self.face2cell

        NFE = self.number_of_edges_of_faces()

        val = np.ones(len(face), dtype=np.bool_)
        I = np.repeat(face2cell[:, 0], NFE)
        cell2edge = coo_matrix((val, (I, face2edge)), shape=(NC, NE), dtype=np.bool_)

        I = np.repeat(face2cell[:, 1], NFE)
        cell2edge += coo_matrix((val, (II, face2edge)), shape=(NC, NE), dtype=np.bool_) 

        return cell2edge.tocsr()

    def cell_to_edge_sign(self):
        pass

    def cell_to_face(self):
        NC = self.NC
        NF = self.NF

        face = self.face
        face2cell = self.face2cell

        val = np.ones((NF,), dtype=np.bool_)
        cell2face = coo_matrix((val, (face2cell[:, 0], range(NF))), shape=(NC, NF), dtype=np.bool_)
        cell2face+= coo_matrix((val, (face2cell[:, 1], range(NF))), shape=(NC, NF), dtype=np.bool_)

        return cell2face.tocsr()

    def cell_to_cell(self):
        NC = self.NC
        face2cell = self.face2cell

        isInFace = (face2cell[:,0] != face2cell[:,1])

        val = np.ones(isInface.sum(), dtype=np.bool_)
        cell2cell = coo_matrix((val, (face2cell[isInface, 0], face2cell[isInFace, 1])), shape=(NC, NC), dtype=np.bool_)
        cell2cell += coo_matrix((val, (face2cell[isInface, 1], face2cell[isInFace, 0])), shape=(NC, NC), dtype=np.bool_)
        return cell2cell.tocsr()

    def face_to_node(self):
        N = self.N
        NF = self.NF

        face = self.face
        NFV = self.number_of_vertices_of_faces()

        I = np.repeat(range(NF), NFV)
        val = np.ones(len(face), dtype=np.bool_)
        face2node = csr_matrix((val, (I, face)), shape=(NF, N), dtype=np.bool_)
        return face2node

    def face_to_edge(self, return_sparse=False):
        NF = self.NF
        NE = self.NE
        face2edge = self.face2edge
        if return_sparse == False:
            return face2edge
        else:
            face = self.face
            NFE = self.number_of_edges_of_faces()
            I = np.repeat(range(NF), NFE)

            val = np.ones(len(face), dtype=np.bool_)
            face2edge = csr_matrix((val, (I, face2edge)), shape=(NF, NE), dtype=np.bool_)
            return face2edge

    def face_to_face(self):
        pass

    def face_to_cell(self, return_sparse=False):
        NF = self.NF
        NC = self.NC
        face2cell = self.face2cell
        if return_sparse == False:
            return face2cell
        else:
            face = self.face
            val = np.ones((NF,), dtype=np.bool_)
            face2cell = coo_matrix((val, (range(NF), face2cell[:, 0])), shape=(NF, NC), dtype=np.bool_)
            face2cell+= coo_matrix((val, (range(NF), face2cell[:, 1])), shape=(NF, NC), dtype=np.bool_)
            return face2cell.tocsr()

    def edge_to_node(self, return_sparse=False):
        N = self.N
        NE = self.NE
        edge = self.edge
        if return_sparse == False:
            return edge
        else:
            val = np.ones(NE, dtype=np.bool_)
            edge2node = coo_matrix((val, (range(NE), edge[:,0])), shape=(NE, N), dtype=np.bool_)
            edge2node+= coo_matrix((val, (range(NE), edge[:,1])), shape=(NE, N), dtype=np.bool_)
            return edge2node.tocsr()

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.transpose()

    def edge_to_face(self):
        NE = self.NE
        NF = self.NF

        face = self.face
        face2edge = self.face2edge
        NFE = self.number_of_edges_of_faces()
 
        J = np.repeat(range(NF), NFE) 
        val = np.ones(len(face), dtype=np.bool_) 
        edge2face = coo_matrix((val, (face2edge, J)), shape=(NE, NF), dtype=np.bool_)
        return edge2face.tocsr()

    def edge_to_cell(self):
        NE = self.NE
        NC = self.NC
        NF = self.NF

        face = self.face
        face2edge = self.face2edge
        face2cell = self.face2cell

        NFE = self.number_of_edges_of_faces()

        J = np.repeat(face2cell[:, 0], NFE)
        val = np.ones(len(face), dtype=np.bool_)
        edge2cell = coo_matirx((val, (face2edge, J)), shape=(NE, NC), dtype=np.bool_)

        J = np.repeat(face2cell[:, 1], NFE)
        edge2cell += coo_matrix((val, (face2edge, J)), shape=(NE, NC), dtype=np.bool_)

        return edge2cell.tocsr()
    
    def node_to_node(self):
        N = self.N
        NE = self.NE
        edge = self.edge
        I = edge.flatten()
        J = edge[:,[1,0]].flatten()
        val = np.ones((2*NE,), dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(N, N),dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        N = self.N
        NE = self.NE
        
        edge = self.edge
        I = edge.flatten()
        J = np.repeat(range(NE), 2)
        val = np.ones(NE, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(N, NE), dtype=np.bool_)
        return node2edge

    def node_to_face(self):
        N = self.N
        NF = self.NF

        face = self.face
        NFV = self.number_of_vertices_of_faces()

        J = np.repeat(range(NF), NFV)
        val = np.ones(len(face), dtype=np.bool_)
        node2face = csr_matrix((val, (face, J)), shape=(N, NF), dtype=np.bool_)
        return node2face


    def node_to_cell(self, cell):
        N = self.N
        NF = self.NF
        NC = self.NC

        face = self.face
        face2cell = self.face2cell
        NFV = self.number_of_vertices_of_faces()

        J = np.repeat(face2cell[:, 0], NFV)
        val = np.ones(len(face), dtype=np.bool_)
        node2cell = coo_matrix((val, (face, J)), shape=(N, NC), dtype=np.bool_)

        J = np.repeat(face2cell[:, 1], NFV)
        node2cell+= coo_matrix((val, (face, J)), shape=(N, NC), dtype=np.bool_)

        return node2cell.tocsr()

    def boundary_node_flag(self):
        N = self.N

        face = self.face

        isBdFace = self.boundary_face_flag() 

        NFV = self.number_of_vertices_of_faces()

        isFaceBdPoint = np.repeat(isBdFace, NFV)

        isBdNode = np.zeros(N, dtype=np.bool_)
        isBdnode[face[isFaceBdNode]] = True
        return isBdNode

    def boundary_edge_flag(self):
        NE = self.NE

        faceLocation = self.faceLocation
        face2edge = self.face2edge

        isBdFace = self.boundary_face_flag()
        NFE = self.number_of_edges_of_faces() 
        isFaceBdEdge = np.repeat(isBdFace, NFE)
        isBdEdge = np.zeros(NE, dtype=np.bool_)
        isBdEdge[face2edge[isFaceBdEdge]] = True
        return isBdEdge

    def boundary_face_flag(self):
        face2cell = self.face2cell
        isBdFace = (face2cell[:,0] == face2cell[:,1])
        return isBdFace

    def boundary_cell_flag(self):
        NC = self.NC
        face2cell = self.face2cell
        isBdFace = self.boundary_face_flag()

        isBdCell = np.zeros(NC, dtype=np.bool_)
        isBdCell[face2cell[isBdFace, 0]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdNode = self.boundary_node_flag()
        idx, = np.nonzero(isBdNode)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_face_index(self):
        isBdFace = self.boundary_face_flag()
        idx, = np.nonzero(isBdFace)
        return idx

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx
