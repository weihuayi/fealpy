import taichi as ti
import numpy as np

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
    i1[j] = np.arange(NEC*NC, dtype=self.itype)

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

    @ti.kernel
    def grad_lambda(self, grad: ti.template()):
        """
        Note:
        """
        for c in range(cell.shape[0]):
            x0 = node[cell[c, 0], 0]
            y0 = node[cell[c, 0], 1]

            x1 = node[cell[c, 1], 0]
            y1 = node[cell[c, 1], 1]

            x2 = node[cell[c, 2], 0]
            y2 = node[cell[c, 2], 1]

            l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
            grad[c][0, 0] = (y1 - y2)/l
            grad[c][0, 1] = (x2 - x1)/l 
            grad[c][1, 0] = (y2 - y0)/l
            grad[c][1, 1] = (x0 - x2)/l
            grad[c][2, 0] = (y0 - y1)/l
            grad[c][2, 1] = (x1 - x0)/l

    @ti.kernel
    def cell_stiff_matrix(self, S: ti.template()):
        """
        组装三角形网格上的最低次单元刚度矩阵

        TODO:
            考虑曲面三角形网格情形
        """
        for c in range(self.cell.shape[0]):
            x0 = self.node[self.cell[c, 0], 0]
            y0 = self.node[self.cell[c, 0], 1]

            x1 = self.node[self.cell[c, 1], 0]
            y1 = self.node[self.cell[c, 1], 1]

            x2 = self.node[self.cell[c, 2], 0]
            y2 = self.node[self.cell[c, 2], 1]

            l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
            gphi = ti.Matrix([
                [(y1 - y2)/l, (x2 - x1)/l], 
                [(y2 - y0)/l, (x0 - x2)/l],
                [(y0 - y1)/l, (x1 - x0)/l]])
            l *= 0.5
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] + gphi[i, 1]*gphi[j, 1])

    def stiff_matrix(self):
        """
        组装三角形网格上最低次的总体刚度矩阵
        """
        K = 0;
        return K 

    def mass_matrix(self):
        M = 0;
        return M
