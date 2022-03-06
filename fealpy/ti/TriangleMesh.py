import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix

class TriangleMeshDataStructure():
    localFace = ti.Matrix([(1, 2), (2, 0), (0, 1)])
    ccw = ti.Matrix([0, 1, 2])

    NVC = 3
    NVE = 2
    NVF = 2

    NEC = 3
    NFC = 3

def construct_edge(NN, cell):
    """ 
    """
    NC =  cell.shape[0] 
    NEC = 3 
    NVE = 2 

    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    totalEdge = cell[:, localEdge].reshape(-1, NVE)
    _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
            return_index=True,
            return_inverse=True,
            axis=0)
    NE = i0.shape[0]
    edge2cell = np.zeros((NE, 4), dtype=np.int_)

    i1 = np.zeros(NE, dtype=np.int_)
    i1[j] = np.arange(NEC*NC, dtype=np.int_)

    edge2cell[:, 0] = i0//NEC
    edge2cell[:, 1] = i1//NEC
    edge2cell[:, 2] = i0%NEC
    edge2cell[:, 3] = i1%NEC

    edge = totalEdge[i0, :]
    cell2edge = np.reshape(j, (NC, NEC))
    return edge, edge2cell, cell2edge


@ti.data_oriented
class TriangleMesh():
    def __init__(self, node, cell):
        assert cell.shape[-1] == 3

        NN = node.shape[0]
        GD = node.shape[1]
        self.node = ti.field(ti.f64, (NN, GD))
        self.node.from_numpy(node)

        NC = cell.shape[0]
        self.cell = ti.field(ti.i32, shape=(NC, 3))
        self.cell.from_numpy(cell)

        edge, edge2cell, cell2edge = construct_edge(NN, cell)
        NE = edge.shape[0]

        self.edge = ti.field(ti.i32, shape=(NE, 2))
        self.edge.from_numpy(edge)

        self.edge2cell = ti.field(ti.i32, shape=(NE, 4))
        self.edge2cell.from_numpy(edge2cell)

        self.cell2edge = ti.field(ti.i32, shape=(NC, 3))
        self.cell2edge.from_numpy(cell2edge)


    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_cells(self):
        return self.cell.shape[0]

    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.cell
        elif etype in {'edge', 'face', 1}:
            return self.edge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`etype` is wrong!")

    @ti.func
    def cell_measure(self, i: ti.i32) -> ti.f64:
        x0 = self.node[self.cell[i, 0], 0]
        y0 = self.node[self.cell[i, 0], 1]

        x1 = self.node[self.cell[i, 1], 0]
        y1 = self.node[self.cell[i, 1], 1]

        x2 = self.node[self.cell[i, 2], 0]
        y2 = self.node[self.cell[i, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        l *= 0.5
        return l

    @ti.func
    def grad_lambda(self, i: ti.i32) -> (ti.types.matrix(3, 2, ti.f64), ti.f64):
        """
        计算第 i 个单元上重心坐标函数的梯度，以及单元的面积
        """
        x0 = self.node[self.cell[i, 0], 0]
        y0 = self.node[self.cell[i, 0], 1]

        x1 = self.node[self.cell[i, 1], 0]
        y1 = self.node[self.cell[i, 1], 1]

        x2 = self.node[self.cell[i, 2], 0]
        y2 = self.node[self.cell[i, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 

        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])

        l *= 0.5
        return gphi, l

    @ti.func
    def surface_grad_lambda(self, i: ti.i32) -> (ti.types.matrix(3, 2, ti.f64), ti.f64):
        """
        计算第 i 个单元上重心坐标函数的梯度，以及单元的面积
        """
        x0 = self.node[self.cell[i, 0], 0]
        y0 = self.node[self.cell[i, 0], 1]
        z0 = self.node[self.cell[i, 0], 2]

        x1 = self.node[self.cell[i, 1], 0]
        y1 = self.node[self.cell[i, 1], 1]
        z1 = self.node[self.cell[i, 0], 2]

        x2 = self.node[self.cell[i, 2], 0]
        y2 = self.node[self.cell[i, 2], 1]
        z2 = self.node[self.cell[i, 0], 2]

        gphi = ti.Matrix(3, 3, ti.f64)
        #TODO:完善代码
        return grad, l

    @ti.kernel
    def cell_stiff_matrices(self, S: ti.template()):
        """
        计算网格上的所有单元刚度矩阵
        """
        for c in range(self.cell.shape[0]):
            gphi, l = self.grad_lambda(c) 

            S[c, 0, 0] = l*(gphi[0, 0]*gphi[0, 0] + gphi[0, 1]*gphi[0, 1])
            S[c, 0, 1] = l*(gphi[0, 0]*gphi[1, 0] + gphi[0, 1]*gphi[1, 1])
            S[c, 0, 2] = l*(gphi[0, 0]*gphi[2, 0] + gphi[0, 1]*gphi[2, 1])

            S[c, 1, 0] = S[c, 0, 1]
            S[c, 1, 1] = l*(gphi[1, 0]*gphi[1, 0] + gphi[1, 1]*gphi[1, 1])
            S[c, 1, 2] = l*(gphi[1, 0]*gphi[2, 0] + gphi[1, 1]*gphi[2, 1])

            S[c, 2, 0] = S[c, 0, 2]
            S[c, 2, 1] = S[c, 1, 2]
            S[c, 2, 2] = l*(gphi[2, 0]*gphi[2, 0] + gphi[2, 1]*gphi[2, 1])

    @ti.kernel
    def cell_mass_matrices(self, S: ti.template()):
        """
        计算网格上的所有单元质量矩阵
        """
        for c in range(cell.shape[0]):

            l = self.cell_measure(c)
            c0 = l/6.0
            c1 = l/12.0

            S[c, 0, 0] = c0 
            S[c, 0, 1] = c1
            S[c, 0, 2] = c1

            S[c, 1, 0] = c1 
            S[c, 1, 1] = c0  
            S[c, 1, 2] = c1

            S[c, 2, 0] = c1 
            S[c, 2, 1] = c1 
            S[c, 2, 2] = c0 

    def stiff_matrix(self, c=None):
        """
        组装总体刚度矩阵
        """
        NN = self.node.shape[0]
        NC = self.cell.shape[0]

        K = ti.field(ti.f64, (NC, 3, 3))
        self.cell_stiff_matrices(K)

        M = K.to_numpy()
        if c is not None:
            M *= c # 目前假设 c 为一常数

        cell = self.cell.to_numpy()
        I = np.broadcast_to(cell[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell[:, None, :], shape=M.shape)
        M = csr_matrix((K.to_numpy().flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def mass_matrix(self, c=None):
        """
        组装总体质量矩阵
        """
        NC = cell.shape[0]

        K = ti.field(ti.f64, (NC, 3, 3))
        self.cell_mass_matrices(K)

        M = K.to_numpy()
        if c is not None:
            M *= c

        I = np.broadcast_to(cell[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell[:, None, :], shape=M.shape)

        NN = node.shape[0]
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    @ti.kernel
    def surface_cell_mass_matrix(self, S: ti.template()):
        """
        组装曲面三角形网格上的最低次单元刚度矩阵， 
        这里的曲面是指三角形网格的节点几何维数为 3
        """
        pass

