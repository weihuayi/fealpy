import numpy as np

class StructureHexMeshDataStructure():
    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (3, 2), (5, 4), (6, 7),
        (2, 0), (1, 3), (4, 6), (7, 5),
        (0, 4), (5, 1), (6, 2), (3, 7)])
    localFace = np.array([
        (0, 2, 6, 4), (1, 5, 7, 3),  # bottom and top faces
        (0, 4, 5, 1), (2, 3, 7, 6),  # left and right faces
        (0, 1, 3, 2), (4, 6, 7, 5)])  # front and back faces
    localFace2edge = np.array([
        (4, 10, 6, 8), (5, 9, 7, 11),
        (0, 8, 2, 9), (1, 11, 3, 10),
        (0, 5, 1, 2), (2, 6, 3, 7)])
    # localEdge = np.array([
    #     (0, 1), (1, 2), (2, 3), (3, 0),
    #     (0, 4), (1, 5), (2, 6), (3, 7),
    #     (4, 5), (5, 6), (6, 7), (7, 4)])
    # localFace = np.array([
    #     (0, 3, 2, 1), (4, 5, 6, 7),  # bottom and top faces
    #     (0, 4, 7, 3), (1, 2, 6, 5),  # left and right faces
    #     (0, 1, 5, 4), (2, 3, 7, 6)])  # front and back faces
    # localFace2edge = np.array([
    #     (0, 1, 2, 3), (8, 9, 10, 11),
    #     (4, 11, 7, 3), (1, 6, 9, 5),
    #     (0, 5, 8, 4), (2, 7, 10, 6)])
    V = 8
    E = 12
    F = 6

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.NN = (nx + 1) * (ny + 1) * (nz + 1)
        self.NE = (nx + 1) * (ny + 1) * nz + (nx + 1) * ny * (nz + 1) + nx * (ny + 1) * (nz + 1)
        self.NF = nx * ny * (nz + 1) + nx * (ny + 1) * nz + (nx + 1) * ny * nz
        self.NC = nx * ny * nz

    @property
    def cell(self):
        NN = self.NN
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NC = self.NC
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int)
        nyz = (ny + 1) * (nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1
        return cell

    @property
    def face(self):
        NN = self.NN
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        face = np.zeros((NF, 4), dtype=np.int)

        NF0 = 0
        NF1 = nx * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:-1, :-1, :]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 2] = face[NF0:NF1, 1] + nz + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 0] + nz + 1
        face[0:nx * ny, :] = face[0:nx * ny, [0, 3, 2, 1]]

        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 2] = face[NF0:NF1, 1] + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 0] + 1
        face[(NF1 - nx * ny):NF1, :] = face[(NF1 - nx * ny):NF1, [1, 0, 3, 2]]

        NF0 = NF1
        NF1 += (nx + 1) * ny * nz
        c = idx[:, :-1, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + nz + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 1] + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 0] + 1
        face[NF0:NF0 + ny * nz, :] = face[NF0:NF0 + ny * nz, [0, 3, 2, 1]]
        return face

    @property
    def face2cell(self):
        NN = self.NN
        NF = self.NF
        NC = self.NC

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NC).reshape(nx, ny, nz)
        face2cell = np.zeros((NF, 4), dtype=np.int)
        # x direction
        NF0 = 0
        NF1 = ny * nz
        face2cell[NF0:NF1, 0] = idx[0].flatten()
        face2cell[NF0:NF1, 1] = idx[0].flatten()
        face2cell[NF0:NF1, 2:4] = 2

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = idx.flatten()
        face2cell[NF0:NF1, 2] = 3
        face2cell[NF0:NF1 - ny * nz, 1] = idx[1:].flatten()
        face2cell[NF0:NF1 - ny * nz, 3] = 2
        face2cell[NF1 - ny * nz:NF1, 1] = idx[-1].flatten()
        face2cell[NF1 - ny * nz:NF1, 3] = 3

        # y direction
        c = np.transpose(idx, (1, 2, 0))
        NF0 = NF1
        NF1 += nx * nz
        face2cell[NF0:NF1, 0] = c[0].flatten()
        face2cell[NF0:NF1, 1] = c[0].flatten()
        face2cell[NF0:NF1, 2:4] = 4

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = c.flatten()
        face2cell[NF0:NF1, 2] = 5
        face2cell[NF0:NF1 - nx * nz, 1] = c[1:].flatten()
        face2cell[NF0:NF1 - nx * nz, 3] = 4
        face2cell[NF1 - nx * nz:NF1, 1] = c[-1].flatten()
        face2cell[NF1 - nx * nz:NF1, 3] = 5

        # z direction
        c = np.transpose(idx, (2, 0, 1))
        NF0 = NF1
        NF1 += nx * ny
        face2cell[NF0:NF1, 0] = c[0].flatten()
        face2cell[NF0:NF1, 1] = c[0].flatten()
        face2cell[NF0:NF1, 2:4] = 0

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = c.flatten()
        face2cell[NF0:NF1, 2] = 1
        face2cell[NF0:NF1 - nx * ny, 1] = c[1:].flatten()
        face2cell[NF0:NF1 - nx * ny, 3] = 0
        face2cell[NF1 - nx * ny:NF1, 1] = c[-1].flatten()
        face2cell[NF1 - nx * ny:NF1, 3] = 1

        return face2cell

    @property
    def edge(self):
        NN = self.NN
        NE = self.NE

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        edge = np.zeros((NE, 2), dtype=np.int)

        NE0 = 0
        NE1 = (nx + 1) * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:, :, :-1]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + 1
        edge[0:(ny + 1) * nz, :] = edge[0:(ny + 1) * nz, [0, 1]]

        NE0 = NE1
        NE1 += (nx + 1) * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:, :-1, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + nz + 1
        edge[(NE1 - (ny + 1) * nz):NE1, :] = edge[(NE1 - (ny + 1) * nz):NE1, [0, 1]]

        NE0 = NE1
        NE1 += (nx + 1) * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + (ny + 1) * (nz + 1)
        edge[NE0:NE0 + (ny + 1) * (nz + 1), :] = edge[NE0:NE0 + (ny + 1) * (nz + 1), [0, 1]]
        return edge

    @property
    def cell2edge(self):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        idx = range(1, NE + 1)
        p2p = csr_matrix((idx, (edge[:, 0], edge[:, 1])), shape=(NN, NN),
                         dtype=np.int)
        totalEdge = self.total_edge()
        cell2edge = np.asarray(p2p[totalEdge[:, 0], totalEdge[:, 1]]).reshape(-1, 12)
        return cell2edge - 1

    def total_edge(self):
        NC = self.NC
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, localEdge.shape[1])
        return np.sort(totalEdge, axis=1)

    def cell_to_node(self):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V * NC, dtype=np.bool)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool)
        return cell2node

    def cell_to_edge(self, sparse=False):
        """ The neighbor information of cell to edge
        """
        if sparse == False:
            return self.cell2edge
        else:
            NC = self.NC
            NE = self.NE
            cell2edge = coo_matrix((NC, NE), dtype=np.bool)
            E = self.E
            I = np.repeat(range(NC), E)
            val = np.ones(E * NC, dtype=np.bool)
            cell2edge = csr_matrix((val, (I, self.cell2edge.flatten())), shape=(NC, NE), dtype=np.bool)
            return cell2edge

    def cell_to_edge_sign(self, cell):
        NC = self.NC
        E = self.E
        cell2edgeSign = np.zeros((NC, E), dtype=np.bool)
        localEdge = self.localEdge
        for i, (j, k) in zip(range(E), localEdge):
            cell2edgeSign[:, i] = cell[:, j] < cell[:, k]
        return cell2edgeSign

    def cell_to_face(self, sparse=False):
        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if sparse == False:
            F = self.F
            cell2face = np.zeros((NC, F), dtype=np.int)
            cell2face[face2cell[:, 0], face2cell[:, 2]] = range(NF)
            cell2face[face2cell[:, 1], face2cell[:, 3]] = range(NF)
            return cell2face
        else:
            val = np.ones((2 * NF,), dtype=np.bool)
            I = face2cell[:, [0, 1]].flatten()
            J = np.repeat(range(NF), 2)
            cell2face = csr_matrix((val, (I, J)), shape=(NC, NF), dtype=np.bool)
            return cell2face

    def cell_to_cell(self, return_sparse=False,
                     return_boundary=True, return_array=False):
        """ Get the adjacency information of cells
        """
        if return_array:
            return_sparse = False
            return_boundary = False

        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if (return_sparse == False) & (return_array == False):
            F = self.F
            cell2cell = np.zeros((NC, F), dtype=np.int)
            cell2cell[face2cell[:, 0], face2cell[:, 2]] = face2cell[:, 1]
            cell2cell[face2cell[:, 1], face2cell[:, 3]] = face2cell[:, 0]
            return cell2cell

        val = np.ones((NF,), dtype=np.bool)
        if return_boundary:
            cell2cell = coo_matrix(
                (val, (face2cell[:, 0], face2cell[:, 1])),
                shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                (val, (face2cell[:, 1], face2cell[:, 0])),
                shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        else:
            isInFace = (face2cell[:, 0] != face2cell[:, 1])
            cell2cell = coo_matrix(
                (val[isInFace], (face2cell[isInFace, 0], face2cell[isInFace, 1])),
                shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                (val[isInFace], (face2cell[isInFace, 1], face2cell[isInFace, 0])),
                shape=(NC, NC), dtype=np.bool)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC + 1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    def face_to_node(self, return_sparse=False):

        face = self.face
        FE = self.localFace.shape[1]
        if return_sparse == False:
            return face
        else:
            N = self.N
            NF = self.NF
            I = np.repeat(range(NF), FE)
            val = np.ones(FE * NF, dtype=np.bool)
            face2node = csr_matrix((val, (I, face)), shape=(NF, N), dtype=np.bool)
            return face2node

    def face_to_edge(self, return_sparse=False):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        FE = localFace2edge.shape[1]
        face2edge = cell2edge[face2cell[:, [0]], localFace2edge[face2cell[:, 2]]]
        if return_sparse == False:
            return face2edge
        else:
            NF = self.NF
            NE = self.NE
            I = np.repeat(range(NF), FE)
            J = face2edge.flatten()
            val = np.ones(FE * NF, dtype=np.bool)
            f2e = csr_matrix((val, (I, J)), shape=(NF, NE), dtype=np.bool)
            return f2e

    def face_to_face(self):
        face2edge = self.face_to_edge()
        return face2edge * face2edge.transpose()

    def face_to_cell(self, return_sparse=False):
        if return_sparse == False:
            return self.face2cell
        else:
            NC = self.NC
            NF = self.NF
            I = np.repeat(range(NF), 2)
            J = self.face2cell[:, [0, 1]].flatten()
            val = np.ones(2 * NF, dtype=np.bool)
            face2cell = csr_matrix((val, (I, J)), shape=(NF, NC), dtype=np.bool)
            return face2cell

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        if return_sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flatten()
            val = np.ones(2 * NE, dtype=np.bool)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
            return edge2node

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node * edge2node.transpose()

    def edge_to_face(self):
        NF = self.NF
        NE = self.NE
        face2edge = self.face_to_edge()
        FE = face2edge.shape[1]
        I = face2edge.flatten()
        J = np.repeat(range(NF), FE)
        val = np.ones(FE * NF, dtype=np.bool)
        edge2face = csr_matrix((val, (I, J)), shape=(NE, NF), dtype=np.bool)
        return edge2face

    def edge_to_cell(self, localidx=False):
        NC = self.NC
        NE = self.NE
        cell2edge = self.cell2edge
        I = cell2edge.flatten()
        E = self.E
        J = np.repeat(range(NC), E)
        val = np.ones(E * NC, dtype=np.bool)
        edge2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool)
        return edge2cell

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flatten()
        J = edge[:, [1, 0]].flatten()
        val = np.ones((2 * NE,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE

        edge = self.edge
        I = edge.flatten()
        J = np.repeat(range(NE), 2)
        val = np.ones(2 * NE, dtype=np.bool)
        node2edge = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
        return node2edge

    def node_to_face(self):
        NN = self.NN
        NF = self.NF

        face = self.face
        FV = face.shape[1]

        I = face.flatten()
        J = np.repeat(range(NF), FV)
        val = np.ones(FV * NF, dtype=np.bool)
        node2face = csr_matrix((val, (I, J)), shape=(NF, NN), dtype=np.bool)
        return node2face

    def node_to_cell(self, return_local_index=False):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flatten()
        J = np.repeat(range(NC), V)

        if return_local_index == True:
            val = ranges(V * np.ones(NC, dtype=np.int), start=1)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int)
        else:
            val = np.ones(V * NC, dtype=np.bool)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        return node2cell

    def boundary_node_flag(self):
        NN = self.NN
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool)
        isBdPoint[face[isBdFace, :]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        NE = self.NE
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = np.zeros((NE,), dtype=np.bool)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge

    def boundary_face_flag(self):
        NF = self.NF
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.NC
        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = np.zeros((NC,), dtype=np.bool)
        isBdCell[face2cell[isBdFace, 0]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_face_index(self):
        isBdFace = self.boundary_face_flag()
        idx, = np.nonzero(isBdFace)
        return idx

    def z_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange((nx+1) * (ny+1) * nz)

    def y_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange((nx+1) * (ny+1) * nz, (nx+1) * (ny+1) * nz + (nx+1) * ny * (nz+1))

    def x_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        return np.arange((nx+1) * (ny+1) * nz + (nx+1) * ny * (nz+1), NE)

    def z_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isZDEdge = np.zeros(NE, dtype=np.bool)
        isZDEdge[:(nx + 1) * (ny + 1) * nz] = True
        return isZDEdge

    def y_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isYDEdge = np.zeros(NE, dtype=np.bool)
        isYDEdge[(nx + 1) * (ny + 1) * nz:-nx * (ny + 1) * (nz + 1)] = True
        return isYDEdge

    def x_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isXDEdge = np.zeros(NE, dtype=np.bool)
        isXDEdge[-nx * (ny + 1) * (nz + 1):] = True
        return isXDEdge

    def z_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange(nx * ny * (nz + 1))

    def y_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange(nx * ny * (nz + 1), nx * ny * (nz + 1) + nx * (ny + 1) * nz)

    def x_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        return np.arange(nx * ny * (nz + 1) + nx * (ny + 1) * nz, NF)

    def z_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isXDFace = np.zeros(NF, dtype=np.bool)
        isXDFace[:nx * ny * (nz + 1)] = True
        return isXDFace

    def y_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isYDFace = np.zeros(NF, dtype=np.bool)
        isYDFace[nx * ny * (nz + 1):-(nx + 1) * ny * nz] = True
        return isYDFace

    def x_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isZDFace = np.zeros(NF, dtype=np.bool)
        isZDFace[-(nx + 1) * ny * nz:] = True
        return isZDFace

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx
