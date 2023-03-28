import numpy as np

class StructureMesh3dDataStructure():
    # The following local data structure should be class properties
    cw = np.array([0, 1, 3, 2])
    ccw = np.array([0, 2, 3, 1])
    localEdge = np.array([
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7)])
    localFace = np.array([
        (0, 1, 2, 3), (4, 5, 6, 7),  # left and right faces
        (0, 1, 4, 5), (2, 3, 6, 7),  # front and back faces
        (0, 2, 4, 6), (1, 3, 5, 7)])  # bottom and top faces
    localFace2edge = np.array([
        (4, 5, 8, 9), (6, 7, 10, 11),
        (0, 1, 8, 10), (2, 3, 9, 11),
        (0, 2, 4, 6), (1, 3, 5, 7)])

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
        """
        @brief 生成网格中所有的单元
        """
        NN = self.NN
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NC = self.NC
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int_)
        nyz = (ny + 1) * (nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + 1
        cell[:, 2] = cell[:, 0] + nz + 1
        cell[:, 3] = cell[:, 2] + 1
        cell[:, 4] = cell[:, 0] + nyz
        cell[:, 5] = cell[:, 4] + 1
        cell[:, 6] = cell[:, 2] + nyz
        cell[:, 7] = cell[:, 6] + 1
        return cell

    @property
    def face(self):
        """
        @brief 生成网格中所有的面
        """
        NN = self.NN
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        face = np.zeros((NF, 4), dtype=np.int_)

        NF0 = 0
        NF1 = (nx + 1) * ny * nz
        c = idx[:, :-1, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 0] + nz + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 2] + 1
        face[NF0:NF0 + ny * nz, :] = face[NF0:NF0 + ny * nz, [1, 0, 3, 2]]

        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 3] = face[NF0:NF1, 2] + 1
        NF2 = NF0 + ny * nz
        N = nz * (ny + 1)
        idx1 = np.zeros((nx, nz), dtype=np.int_)
        idx1 = np.arange(NF2, NF2 + nz)
        idx1 = idx1 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx1 = idx1.flatten()
        face[idx1] = face[idx1][:, [1, 0, 3, 2]]

        NF0 = NF1
        NF1 += nx * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:-1, :-1, :]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + nz + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 3] = face[NF0:NF1, 2] + nz + 1
        N = ny * (nz + 1)
        idx2 = np.zeros((nx, ny), dtype=np.int_)
        idx2 = np.arange(NF0, NF0 + ny * (nz + 1), nz + 1)
        idx2 = idx2 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx2 = idx2.flatten()
        face[idx2] = face[idx2][:, [1, 0, 3, 2]]

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
        face2cell = np.zeros((NF, 4), dtype=np.int_)

        # x direction
        NF0 = 0
        NF1 = ny * nz
        face2cell[NF0:NF1, 0] = idx[0].flatten()
        face2cell[NF0:NF1, 1] = idx[0].flatten()
        face2cell[NF0:NF1, 2:4] = 0

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = idx.flatten()
        face2cell[NF0:NF1, 2] = 1
        face2cell[NF0:NF1 - ny * nz, 1] = idx[1:].flatten()
        face2cell[NF0:NF1 - ny * nz, 3] = 0
        face2cell[NF1 - ny * nz:NF1, 1] = idx[-1].flatten()
        face2cell[NF1 - ny * nz:NF1, 3] = 1

        # y direction
        c = np.transpose(idx, (1, 0, 2))
        NF0 = NF1
        NF1 += nx * nz
        face2cell[NF0:NF1, 0] = c[0].flatten()
        face2cell[NF0:NF1, 1] = c[0].flatten()
        face2cell[NF0:NF1, 2:4] = 2

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = c.flatten()
        face2cell[NF0:NF1, 2] = 3
        face2cell[NF0:NF1 - nx * nz, 1] = c[1:].flatten()
        face2cell[NF0:NF1 - nx * nz, 3] = 2
        face2cell[NF1 - nx * nz:NF1, 1] = c[-1].flatten()
        face2cell[NF1 - nx * nz:NF1, 3] = 3

        # z direction
        c = np.transpose(idx, (2, 0, 1))
        NF0 = NF1
        NF1 += nx * ny
        face2cell[NF0:NF1, 0] = c[0].flatten()
        face2cell[NF0:NF1, 1] = c[0].flatten()
        face2cell[NF0:NF1, 2:4] = 4

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = c.flatten()
        face2cell[NF0:NF1, 2] = 5
        face2cell[NF0:NF1 - nx * ny, 1] = c[1:].flatten()
        face2cell[NF0:NF1 - nx * ny, 3] = 4
        face2cell[NF1 - nx * ny:NF1, 1] = c[-1].flatten()
        face2cell[NF1 - nx * ny:NF1, 3] = 5

        return face2cell

    @property
    def edge(self):
        """
        @brief 生成网格中所有的边
        """
        NN = self.NN
        NE = self.NE

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        edge = np.zeros((NE, 2), dtype=np.int_)

        NE0 = 0
        NE1 = nx * (ny + 1) * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + (ny + 1) * (nz + 1)

        NE0 = NE1
        NE1 += (nx + 1) * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:, :-1, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + nz + 1

        NE0 = NE1
        NE1 += (nx + 1) * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:, :, :-1]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + 1

        return edge

    def cell_to_edge(self):
        """
        The neighbor information of cell to edge
        @brief 单元和边的邻接关系, 储存每个单元相邻的边的编号
        """
        NC = self.NC
        NE = self.NE

        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2edge = np.zeros((NC, 12), dtype=np.int_)

        idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
        cell2edge[:, 0] = idx0[:, :-1, :-1].flatten()
        cell2edge[:, 1] = idx0[:, :-1, 1:].flatten()
        cell2edge[:, 2] = idx0[:, 1:, :-1].flatten()
        cell2edge[:, 3] = idx0[:, 1:, 1:].flatten()

        NE0 = nx * (ny + 1) * (nz + 1)
        idx1 = np.arange((nx + 1) * ny * (nz + 1)).reshape(nx + 1, ny, nz + 1)
        cell2edge[:, 4] = (NE0 + idx1[:-1, :, :-1]).flatten()
        cell2edge[:, 5] = (NE0 + idx1[:-1, :, 1:]).flatten()
        cell2edge[:, 6] = (NE0 + idx1[1:, :, :-1]).flatten()
        cell2edge[:, 7] = (NE0 + idx1[1:, :, 1:]).flatten()

        NE1 = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1)
        idx2 = np.arange((nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny + 1, nz)
        cell2edge[:, 8] = (NE1 + idx2[:-1, :-1, :]).flatten()
        cell2edge[:, 9] = (NE1 + idx2[:-1, 1:, :]).flatten()
        cell2edge[:, 10] = (NE1 + idx2[1:, :-1, :]).flatten()
        cell2edge[:, 11] = (NE1 + idx2[1:, 1:, :]).flatten()

        return cell2edge

    def total_edge(self):
        """
        @brief 储存每个单元的所有边的节点编号
        """
        NC = self.NC
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, localEdge.shape[1])
        return np.sort(totalEdge, axis=1)

    def cell_to_node(self):
        """
        @brief 判断单元中的节点, 若单元中有这个节点为 True, 否则为 False
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V * NC, dtype=np.bool_)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool_)
        return cell2node

    def total_face(self):
        """
        @brief 储存每个单元的所有面的节点编号
        """
        NC = self.NC
        cell = self.cell
        localFace = self.localFace
        totalFace = cell[:, localFace].reshape(-1, localFace.shape[1])
        return np.sort(totalFace, axis=1)

    def cell_to_face(self, sparse=False):
        """
        @brief 单元和面的邻接关系, 储存每个单元相邻的六个面的编号
        """
        NC = self.NC
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2face = np.zeros((NC, 6), dtype=np.int_)

        # x direction
        idx0 = np.arange((nx + 1) * ny * nz).reshape(nx + 1, ny, nz)
        cell2face[:, 0] = idx0[:-1, :, :].flatten()
        cell2face[:, 1] = idx0[1:, :, :].flatten()

        # y direction
        NE0 = (nx + 1) * ny * nz
        idx1 = np.arange(nx * (ny + 1) * nz).reshape(nx, ny + 1, nz)
        cell2face[:, 2] = (NE0 + idx1[:, :-1, :]).flatten()
        cell2face[:, 3] = (NE0 + idx1[:, 1:, :]).flatten()

        # z direction
        NE1 = (nx + 1) * ny * nz + nx * (ny + 1) * nz
        idx2 = np.arange(nx * ny * (nz + 1)).reshape(nx, ny, nz + 1)
        cell2face[:, 4] = (NE1 + idx2[:, :, :-1]).flatten()
        cell2face[:, 5] = (NE1 + idx2[:, :, 1:]).flatten()

        return cell2face

    def cell_to_cell(self, return_sparse=False,
                     return_boundary=True, return_array=False):
        """
        Get the adjacency information of cells
        @brief 单元和单元的邻接关系, 储存每个单元相邻的六个单元的编号
        """
        NN = self.NN
        NC = self.NC

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NC).reshape(nx, ny, nz)
        cell2cell = np.zeros((NC, 6), dtype=np.int_)

        # x direction
        NE0 = 0
        NE1 = ny * nz
        NE2 = nx * ny * nz
        cell2cell[NE0: NE1, 0] = idx[0, :, :].flatten()
        cell2cell[NE1: NE2, 0] = idx[:-1, :, :].flatten()
        cell2cell[NE0: NE2 - NE1, 1] = idx[1:, :, :].flatten()
        cell2cell[NE2 - NE1: NE2, 1] = idx[-1, :, :].flatten()

        # y direction
        N = ny * nz
        idx0 = np.zeros((nx, nz), dtype=np.int_)
        idx0 = np.arange(NE0, NE0 + nz)
        idx0 = idx0 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx0 = idx0.flatten()

        NE1 = NE0 + nz * (ny - 1)
        idx1 = np.zeros((nx, nz), dtype=np.int_)
        idx1 = np.arange(NE1, NE1 + nz)
        idx1 = idx1 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx1 = idx1.flatten()

        cell2cell[idx0, 2] = idx0
        ii = np.setdiff1d(idx.flatten(), idx0)
        cell2cell[ii, 2] = ii - nz

        cell2cell[idx1, 3] = idx1
        ii = np.setdiff1d(idx.flatten(), idx1)
        cell2cell[ii, 3] = ii + nz

        # z direction
        N = ny * nz
        idx2 = np.zeros((nx, ny), dtype=np.int_)
        idx2 = np.arange(NE0, NE0 + N, nz)
        idx2 = idx2 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx2 = idx2.flatten()

        NE1 = NE0 + (nz - 1)
        idx3 = np.zeros((nx, ny), dtype=np.int_)
        idx3 = np.arange(NE1, NE1 + N, nz)
        idx3 = idx3 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx3 = idx3.flatten()

        cell2cell[idx2, 4] = idx2
        ii = np.setdiff1d(idx.flatten(), idx2)
        cell2cell[ii, 4] = ii - 1

        cell2cell[idx3, 5] = idx3
        ii = np.setdiff1d(idx.flatten(), idx3)
        cell2cell[ii, 5] = ii + 1

        return cell2cell

    def face_to_node(self, return_sparse=False):
        """
        @brief 面和节点的邻接关系, 储存每个面相邻的四个节点的编号
        """
        face = self.face
        if return_sparse == False:
            return face
        else:
            N = self.N
            NF = self.NF
            I = np.repeat(range(NF), 4)
            val = np.ones(4 * NF, dtype=np.bool_)
            face2node = csr_matrix((val, (I, face.flat)), shape=(NF, N), dtype=np.bool_)
            return face2node

    def face_to_edge(self, return_sparse=False):
        """
        @brief 面和边的邻接关系, 储存每个面相邻的四个边的编号
        """

        NE = self.NE
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz
        face2edge = np.zeros((NF, 4), dtype=np.int_)

        # x direction
        NE0 = 0
        NE1 = (nx + 1) * ny * nz
        idx0 = np.arange(nx * (ny + 1) * (nz + 1), NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
        face2edge[NE0:NE1, 0] = idx0[:, :, :-1].flatten()
        face2edge[NE0:NE1, 1] = idx0[:, :, 1:].flatten()

        idx1 = np.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
        face2edge[NE0:NE1, 2] = idx1[:, :-1, :].flatten()
        face2edge[NE0:NE1, 3] = idx1[:, 1:, :].flatten()

        # y direction
        NE0 = NE1
        NE1 += nx * (ny + 1) * nz
        idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
        face2edge[NE0:NE1, 0] = idx0[:, :, :-1].flatten()
        face2edge[NE0:NE1, 1] = idx0[:, :, 1:].flatten()

        idx1 = np.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
        face2edge[NE0:NE1, 2] = idx1[:-1, :, :].flatten()
        face2edge[NE0:NE1, 3] = idx1[1:, :, :].flatten()

        # z direction
        NE0 = NE1
        NE1 += nx * ny * (nz + 1)
        idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
        face2edge[NE0:NE1, 0] = idx0[:, :-1, :].flatten()
        face2edge[NE0:NE1, 1] = idx0[:, 1:, :].flatten()

        idx1 = np.arange(nx * (ny + 1) * (nz + 1), NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
        face2edge[NE0:NE1, 2] = idx1[:-1, :, :].flatten()
        face2edge[NE0:NE1, 3] = idx1[1:, :, :].flatten()

        return face2edge

    def face_to_face(self):
        """
        @brief 判断两个面是否相邻，相邻为 True, 否则为 False
        """
        edge2face = self.edge_to_face()
        return edge2face.T * edge2face.transpose().T

    def face_to_cell(self, return_sparse=False):
        """
        @brief 面和单元的邻接关系, 储存每个面相邻的两个单元的编号
        """
        if return_sparse == False:
            return self.face2cell
        else:
            NC = self.NC
            NF = self.NF
            I = np.repeat(range(NF), 2)
            J = self.face2cell[:, [0, 1]].flatten()
            val = np.ones(2 * NF, dtype=np.bool_)
            face2cell = csr_matrix((val, (I, J)), shape=(NF, NC), dtype=np.bool_)
            return face2cell

    def edge_to_node(self, return_sparse=False):
        """
        @brief 边和节点的邻接关系, 储存每个边相邻的两个节点的编号
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        if return_sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flatten()
            val = np.ones(2 * NE, dtype=np.bool_)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool_)
            return edge2node

    def edge_to_edge(self):
        """
        @brief 判断两条边是否相邻，相邻为 True, 否则为 False
        """
        node2edge = self.node_to_edge()
        return node2edge.T * node2edge.transpose().T

    def edge_to_face(self):
        """
        @brief 判断边是否为某面的边，若是则对应位置为 True,否则为 False
        """
        NF = self.NF
        NE = self.NE
        face2edge = self.face_to_edge()
        FE = face2edge.shape[1]
        I = face2edge.flatten()
        J = np.repeat(range(NF), FE)
        val = np.ones(FE * NF, dtype=np.bool_)
        edge2face = csr_matrix((val, (I, J)), shape=(NE, NF), dtype=np.bool_)
        return edge2face

    def edge_to_cell(self, localidx=False):
        """
        @brief 判断边是否为某单元的边，若是则对应位置为 True,否则为 False
        """
        NC = self.NC
        NE = self.NE
        cell2edge = self.cell2edge
        I = cell2edge.flatten()
        E = self.E
        J = np.repeat(range(NC), E)
        val = np.ones(E * NC, dtype=np.bool_)
        edge2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool_)
        return edge2cell

    def node_to_node(self):
        """
        The neighbor information of nodes
        @brief 判断某两个节点是否相邻，若是则对应位置为True，否则为False
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flatten()
        J = edge[:, [1, 0]].flatten()
        val = np.ones((2 * NE,), dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        """
        @brief 判断节点是否为某边的端点，若是则对应位置为 True,否则为 False
        """
        NN = self.NN
        NE = self.NE

        edge = self.edge
        I = edge.flatten()
        J = np.repeat(range(NE), 2)
        val = np.ones(2 * NE, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NE), dtype=np.bool_)
        return node2edge

    def node_to_face(self):
        """
        @brief 判断节点是否为某面的端点，若是则对应位置为 True,否则为 False
        """
        NN = self.NN
        NF = self.NF

        face = self.face
        FV = face.shape[1]

        I = face.flatten()
        J = np.repeat(range(NF), FV)
        val = np.ones(FV * NF, dtype=np.bool_)
        node2face = csr_matrix((val, (I, J)), shape=(NN, NF), dtype=np.bool_)
        return node2face

    def node_to_cell(self, return_local_index=False):
        """
        @brief 判断节点是否为某单元的端点，若是则对应位置为 True,否则为 False
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flatten()
        J = np.repeat(range(NC), V)

        if return_local_index == True:
            val = ranges(V * np.ones(NC, dtype=np.int_), start=1)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int_)
        else:
            val = np.ones(V * NC, dtype=np.bool_)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool_)
        return node2cell

    def boundary_node_flag(self):
        """
        @brief 判断是否为边界点
        """
        NN = self.NN
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool_)
        isBdPoint[face[isBdFace, :]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        """
        @brief 判断边是否为边界边
        """
        NE = self.NE
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = np.zeros((NE,), dtype=np.bool_)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge

    def boundary_face_flag(self):
        """
        @brief 判断单元是否为边界面
        """
        NF = self.NF
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self):
        """
        @brief 判断单元是否为边界单元
        """
        NC = self.NC
        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = np.zeros((NC,), dtype=np.bool_)
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

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx

    def x_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange(nx * (ny + 1) * (nz + 1))

    def y_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange(nx * (ny + 1) * (nz + 1), nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1))

    def z_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        return np.arange(nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1), NE)

    def x_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isXDEdge = np.zeros(NE, dtype=np.bool_)
        isXDEdge[:nx * (ny + 1) * (nz + 1)] = True
        return isXDEdge

    def y_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isYDEdge = np.zeros(NE, dtype=np.bool_)
        isYDEdge[nx * (ny + 1) * (nz + 1):-(nx + 1) * (ny + 1) * nz] = True
        return isYDEdge

    def z_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isZDEdge = np.zeros(NE, dtype=np.bool_)
        isZDEdge[-(nx + 1) * (ny + 1) * nz:] = True
        return isZDEdge

    def x_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange((nx + 1) * ny * nz)

    def y_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange((nx + 1) * ny * nz, (nx + 1) * ny * nz + nx * (ny + 1) * nz)

    def z_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        return np.arange((nx + 1) * ny * nz + nx * (ny + 1) * nz, NF)

    def x_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isZDFace = np.zeros(NF, dtype=np.bool_)
        isZDFace[:(nx + 1) * ny * nz] = True
        return isZDFace

    def y_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isYDFace = np.zeros(NF, dtype=np.bool_)
        isYDFace[(nx + 1) * ny * nz:-nx * ny * (nz + 1)] = True
        return isYDFace

    def z_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isXDFace = np.zeros(NF, dtype=np.bool_)
        isXDFace[-nx * ny * (nz + 1):] = True
        return isXDFace
