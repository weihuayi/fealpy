import numpy as np
from .Mesh2d import Mesh2d, Mesh2dDataStructure


class QuadMeshDataStructure(Mesh2dDataStructure):
    """
    Notice that vertex order of cell in QuadMesh is z-order as following:
        2-------------3
        |             |
        |             |
        |             |
        |             |
        0-------------1
    """
    localEdge = np.array([(0, 2), (1, 3), (0, 1), (2, 3)])
    ccw = np.array([0, 1, 3, 2], dtype=np.int8)
    V = 4
    E = 4
    F = 1

    def __init__(self, NN, cell):
        super(QuadMeshDataStructure, self).__init__(NN, cell)


class QuadMesh(Mesh2d):
    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = QuadMeshDataStructure(NN, cell)

        self.meshtype = 'quad'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}

    def area(self, index=None):
        return self.cell_area(index=index)

    def cell_area(self, index=None):
        node = self.entity('node')
        cell = self.entity('cell')
        if index is None:
            v1 = node[cell[:, 1], :] - node[cell[:, 0], :]
            v2 = node[cell[:, 2], :] - node[cell[:, 0], :]
        else:
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v1, v2)
        a0 = nv/2.0

        if index is None:
            v1 = node[cell[:, 3], :] - node[cell[:, 2], :]
            v2 = node[cell[:, 0], :] - node[cell[:, 2], :]
        else:
            v1 = node[cell[index, 3], :] - node[cell[index, 2], :]
            v2 = node[cell[index, 0], :] - node[cell[index, 2], :]
        nv = np.cross(v1, v2)
        a1 = nv/2.0
        return a0 + a1

    def uniform_refine(self, n=1):
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # Find the cutted edge 
            cell2edge = self.ds.cell_to_edge()
            edgeCenter = self.barycenter(entity='edge')
            cellCenter = self.barycenter(entity='cell')

            edge2center = np.arange(NN, NN+NE)

            cell = self.ds.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(NN + NE, NN + NE + NC).reshape(-1, 1)

            cell = np.zeros((4*NC, 4), dtype=np.int)
            cell[0::4, :] = np.r_['1', cp[0], ep[2], ep[0], cc]
            cell[1::4, :] = np.r_['1', ep[2], cp[1], cc, ep[1]]
            cell[2::4, :] = np.r_['1', ep[0], cc, cp[2], ep[3]]
            cell[3::4, :] = np.r_['1', cc, ep[1], ep[3], cp[3]]

            self.node = np.r_['0', self.node, edgeCenter, cellCenter]
            self.ds.reinit(NN + NE + NC, cell)

    def bc_to_point(self, bc):
        """
        重心坐标的顺序是逆时针方向，而 QuadMesh 单元顶点的编号满足 z-order
 
        2-------------3
        |             |
        |             |
        |             |
        |             |
        0-------------1
        """
        node = self.node
        cell = self.ds.cell
        p = np.einsum('...j, ijk->...ik', bc, node[cell[:, self.ccw]])
        return p
