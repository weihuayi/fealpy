import numpy as np

"""
结构四边形网格的拓扑数据结构

单元节点的排序如下：

1 ------- 3
|         |
|         |
|         |
0 ------- 2

整个网格中实体的排序规则：

* 节点的编号规则，先排 y 方向，再排 x 方向
* 边的编号规则，先排 y 方向，再排 x 方向
* 单元的编号规则，先排 y 方向，再排 x 方向

"""

class StructureMesh2dDataStructure:
    cw = np.array([0, 1, 3, 2])
    ccw = np.array([0, 2, 3, 1])
    localEdge = np.array([(0, 2), (1, 3), (0, 1), (2, 3)])

    V = 4
    E = 4
    F = 1

    def __init__(self, nx, ny, itype):
        self.nx = nx  # x 方向剖分的段数
        self.ny = ny  # y 方向剖分的段数
        self.NN = (nx + 1) * (ny + 1)
        self.NE = ny * (nx + 1) + nx * (ny + 1)
        self.NC = nx * ny
        self.NF = self.NE
        self.itype = itype

    def number_of_nodes_of_cells(self):
        return self.V

    def number_of_edges_of_cells(self):
        return self.E

    def number_of_faces_of_cells(self):
        return self.E

    def number_of_vertices_of_cells(self):
        return self.V

    @property
    def cell(self):
        """
        @brief 生成网格中所有的单元
        """

        nx = self.nx
        ny = self.ny

        NN = self.NN
        NC = self.NC
        cell = np.zeros((NC, 4), dtype=self.itype)
        idx = np.arange(NN).reshape(nx + 1, ny + 1)
        c = idx[:-1, :-1]
        cell[:, 0] = c.flat
        cell[:, 1] = cell[:, 0] + 1
        cell[:, 2] = cell[:, 0] + ny + 1
        cell[:, 3] = cell[:, 2] + 1
        return cell

    @property
    def edge(self):
        """
        @brief 生成网格中所有的边
        """

        nx = self.nx
        ny = self.ny

        NN = self.NN
        NE = self.NE

        idx = np.arange(NN, dtype=self.itype).reshape(nx + 1, ny + 1)
        edge = np.zeros((NE, 2), dtype=self.itype)

        NE0 = 0
        NE1 = nx * (ny + 1)
        edge[NE0:NE1, 0] = idx[:-1, :].flat
        edge[NE0:NE1, 1] = idx[1:, :].flat
        edge[NE0 + ny:NE1:ny + 1, :] = edge[NE0 + ny:NE1:ny + 1, -1::-1]

        NE0 = NE1
        NE1 += ny * (nx + 1)
        edge[NE0:NE1, 0] = idx[:, :-1].flat
        edge[NE0:NE1, 1] = idx[:, 1:].flat
        edge[NE0:NE0 + ny, :] = edge[NE0:NE0 + ny, -1::-1]
        return edge

    @property
    def edge2cell(self):
        """
        @brief 边与单元的邻接关系，储存与每条边相邻的两个单元的信息
        """

        nx = self.nx
        ny = self.ny

        NC = self.NC
        NE = self.NE

        edge2cell = np.zeros((NE, 4), dtype=self.itype)

        idx = np.arange(NC).reshape(nx, ny).T

        # x direction
        idx0 = np.arange(nx * (ny + 1), dtype=self.itype).reshape(nx, ny + 1).T
        # left element
        edge2cell[idx0[:-1], 0] = idx
        edge2cell[idx0[:-1], 2] = 0
        edge2cell[idx0[-1], 0] = idx[-1]
        edge2cell[idx0[-1], 2] = 1

        # right element
        edge2cell[idx0[1:], 1] = idx
        edge2cell[idx0[1:], 3] = 1
        edge2cell[idx0[0], 1] = idx[0]
        edge2cell[idx0[0], 3] = 0

        # y direction
        idx1 = np.arange((nx + 1) * ny, dtype=self.itype).reshape(nx + 1, ny).T
        NE0 = nx * (ny + 1)
        # left element
        edge2cell[NE0 + idx1[:, 1:], 0] = idx
        edge2cell[NE0 + idx1[:, 1:], 2] = 3
        edge2cell[NE0 + idx1[:, 0], 0] = idx[:, 0]
        edge2cell[NE0 + idx1[:, 0], 2] = 2

        # right element
        edge2cell[NE0 + idx1[:, :-1], 1] = idx
        edge2cell[NE0 + idx1[:, :-1], 3] = 2
        edge2cell[NE0 + idx1[:, -1], 1] = idx[:, -1]
        edge2cell[NE0 + idx1[:, -1], 3] = 3

        return edge2cell

    def cell_to_node(self):
        """
        @brief 单元和节点的邻接关系，储存每个单元相邻的节点编号
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V * NC, dtype=np.bool_)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool_)
        return cell2node

    def cell_to_edge(self, sparse=False):
        """
        The neighbor information of cell to edge
        @brief 单元和边的邻接关系，储存每个单元相邻的边的编号
        """
        NC = self.NC
        NE = self.NE

        nx = self.nx
        ny = self.ny

        cell2edge = np.zeros((NC, 4), dtype=np.int)

        idx0 = np.arange(nx * (ny + 1)).reshape(nx, ny + 1)
        cell2edge[:, 0] = idx0[:, :-1].flatten()
        cell2edge[:, 1] = idx0[:, 1:].flatten()

        idx1 = np.arange(nx * (ny + 1), NE).reshape(nx + 1, ny)
        cell2edge[:, 2] = idx1[:-1, :].flatten()
        cell2edge[:, 3] = idx1[1:, :].flatten()

        return cell2edge

    def cell_to_cell(self, return_sparse=False, return_boundary=True, return_array=False):
        """
        Consctruct the neighbor information of cells
        @brief 单元和单元的邻接关系，储存每个单元相邻的单元编号
        """
        NN = self.NN
        NC = self.NC

        nx = self.nx
        ny = self.ny
        idx = np.arange(NC).reshape(nx, ny)
        cell2cell = np.zeros((NC, 4), dtype=np.int)

        # x direction
        NE0 = 0
        NE1 = ny
        NE2 = nx * ny
        cell2cell[NE0: NE1, 0] = idx[0, :].flatten()
        cell2cell[NE1: NE2, 0] = idx[:-1, :].flatten()
        cell2cell[NE0: NE2 - NE1, 1] = idx[1:, :].flatten()
        cell2cell[NE2 - NE1: NE2, 1] = idx[-1, :].flatten()

        # y direction
        idx0 = np.arange(0, nx * ny, ny).reshape(nx, 1)
        idx0 = idx0.flatten()

        idx1 = idx0 + ny - 1
        idx1 = idx1.flatten()

        cell2cell[idx0, 2] = idx0
        ii = np.setdiff1d(idx.flatten(), idx0)
        cell2cell[ii, 2] = ii - 1

        cell2cell[idx1, 3] = idx1
        ii = np.setdiff1d(idx.flatten(), idx1)
        cell2cell[ii, 3] = ii + 1

        return cell2cell

    def edge_to_node(self, sparse=False):
        """
        @brief 边与节点的邻接关系，储存每条边的两个端点的节点编号
        """
        NN = self.NN
        NE = self.NE

        edge = self.edge
        if sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flat
            val = np.ones(2 * NE, dtype=np.bool_)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool_)
            return edge2node

    def edge_to_edge(self, sparse=False):
        """
        @brief 判断两条边是否相邻，相邻为 True, 否则为 False
        """
        node2edge = self.node_to_edge()
        return node2edge.T * node2edge.transpose().T

    def edge_to_cell(self, sparse=False):
        """
        @brief 边与单元的邻接关系，储存与每条边相邻的两个单元的信息
        """
        if sparse == False:
            return self.edge2cell
        else:
            NC = self.NC
            NE = self.NE
            I = np.repeat(range(NF), 2)
            J = self.edge2cell[:, [0, 1]].flatten()
            val = np.ones(2 * NE, dtype=np.bool_)
            face2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool_)
            return face2cell

    def node_to_node(self):
        """
        The neighbor information of nodes
        @brief 判断某两个节点是否相邻，若是则对应位置为True，否则为False
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flat
        J = edge[:, [1, 0]].flat
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

    def node_to_cell(self, localidx=False):
        """
        @brief 判断节点是否位于某单元中，位于则对应位置为True，否则为False
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flatten()
        J = np.repeat(range(NC), V)

        if localidx == True:
            val = ranges(V * np.ones(NC, dtype=np.int), start=1)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int)
        else:
            val = np.ones(V * NC, dtype=np.bool_)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool_)
        return node2cell

    def boundary_node_flag(self):
        """
        @brief 判断是否为边界点
        """
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool_)
        isBdPoint[edge[isBdEdge, :]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        """
        @brief 判断边是否为边界边
        """
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    def boundary_cell_flag(self, bctype=None):
        """
        @brief 判断单元是否为边界单元
        """
        NC = self.NC

        if bctype is None:
            edge2cell = self.edge2cell
            isBdCell = np.zeros((NC,), dtype=np.bool_)
            isBdEdge = self.boundary_edge_flag()
            isBdCell[edge2cell[isBdEdge, 0]] = True

        else:
            cell2cell = self.cell_to_cell()
            isBdCell = cell2cell[:, bctype] == np.arange(NC)
        return isBdCell

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_cell_index(self, bctype=None):
        isBdCell = self.boundary_cell_flag(bctype)
        idx, = np.nonzero(isBdCell)
        return idx

    def x_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        return np.arange(nx * (ny + 1))

    def y_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        return np.arange(nx * (ny + 1), NE)

    def x_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        isXDEdge = np.zeros(NE, dtype=np.bool_)
        isXDEdge[:nx * (ny + 1)] = True
        return isXDEdge

    def y_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        isYDEdge = np.zeros(NE, dtype=np.bool_)
        isYDEdge[nx * (ny + 1):] = True
        return isYDEdge

    def left_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        return np.arange(ny + 1)

    def right_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(NN - ny - 1, NN)

    def bottom_boundary_node__index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(0, NN - ny, ny + 1)

    def up_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(ny, NN, ny + 1)

    def peoriod_matrix(self):
        """
        we can get a matarix under periodic boundary condition
        """
        nx = self.nx
        ny = self.ny
        NN = self.NN
        isPNode = np.zeros(NN, dtype=np.bool_)
        lidx = self.left_boundary_node_index()
        ridx = self.right_boundary_node_index()
        bidx = self.bottom_boundary_node__index()
        uidx = self.up_boundary_node_index()

        isPNode[ridx] = True
        isPNode[uidx] = True
        NC = nx * ny
        # First, we get the inner elements , the left boundary and the lower boundary of the matrix.
        val = np.ones(NC, dtype=np.bool_)
        I = np.arange(NN)[~isPNode]
        J = range(NC)
        C = coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool_)
        # second,  we make the upper boundary equal to the lower boundary.
        val = np.ones(nx, dtype=np.bool_)
        I = np.arange(NN)[uidx[:-1]]
        J = np.arange(0, NC - ny + 1, ny)
        C += coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool_)
        # thrid, we make the right boundary equal to the left boundary.
        val = np.ones(ny + 1, dtype=np.bool_)
        I = np.arange(NN)[ridx]
        J = np.arange(ny + 1)
        J[-1] = 0
        C += coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool_)

        return C
