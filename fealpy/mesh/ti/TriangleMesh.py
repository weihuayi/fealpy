
import numpy as np
import taichi as ti
from scipy.sparse import csr_matrix

def construct_edge(cell):
    """
    """
    NC =  cell.shape[0]
    NEC = 3
    NVE = 2

    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    totalEdge = cell[:, localEdge].reshape(-1, 2)
    _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
            return_index=True,
            return_inverse=True,
            axis=0)
    NE = i0.shape[0]
    edge2cell = np.zeros((NE, 4), dtype=np.int_)

    i1 = np.zeros(NE, dtype=np.int_)
    i1[j] = np.arange(NEC*NC, dtype=np.int_)

    edge2cell[:, 0] = i0//3
    edge2cell[:, 1] = i1//3
    edge2cell[:, 2] = i0%3
    edge2cell[:, 3] = i1%3

    edge = totalEdge[i0, :]
    cell2edge = np.reshape(j, (NC, 3))
    return edge, edge2cell, cell2edge


@ti.data_oriented
class TriangleMesh():
    def __init__(self, node, cell, itype=ti.u32, ftype=ti.f64):
        assert cell.shape[-1] == 3


        NN = node.shape[0]
        GD = node.shape[1]
        self.node = ti.Vector.field(2, dtype=ftype, shape=NN)
        self.node.from_numpy(node)

        NC = cell.shape[0]
        self.cell = ti.field(itype, shape=(NC, 3))
        self.cell.from_numpy(cell)

        self.glambda = ti.field(ftype, shape=(NC, 3, GD))
        self.cellmeasure = ti.field(ftype, shape=(NC, ))

        self.init_grad_lambdas()

        self.itype = itype
        self.ftype = ftype

        self.construct_data_structure(cell)


    def construct_data_structure(self, cell):

        edge, edge2cell, cell2edge = construct_edge(cell)
        NE = edge.shape[0]
        self.edge = ti.field(self.itype, shape=(NE, 2))
        self.edge.from_numpy(edge)

        self.edge2cell = ti.field(self.itype, shape=(NE, 4))
        self.edge2cell.from_numpy(edge2cell)

        NC = self.number_of_cells()
        self.cell2edge = ti.field(self.itype, shape=(NC, 3))
        self.cell2edge.from_numpy(cell2edge)

    def multi_index_matrix(self, p):
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 1] = idx0 - multiIndex[:,2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex

    def number_of_local_interpolation_points(self, p):
        return (p+1)*(p+2)//2

    def number_of_global_interpolation_points(self, p):
        NP = self.number_of_nodes()
        if p > 1:
            NE = self.number_of_edges()
            NP += (p-1)*NE
        if p > 2:
            NC = self.number_of_cells()
            NP += (p-2)*(p-1)*NC//2
        return NP

    @ti.kernel
    def init_grad_lambdas(self):
        """
        @brief 初始化网格中每个单元上重心坐标函数的梯度，以及单元的面积
        """
        for i in range(self.cell.shape[0]):
            x0 = self.node[self.cell[i, 0]][0]
            y0 = self.node[self.cell[i, 0]][1]

            x1 = self.node[self.cell[i, 1]][0]
            y1 = self.node[self.cell[i, 1]][1]

            x2 = self.node[self.cell[i, 2]][0]
            y2 = self.node[self.cell[i, 2]][1]

            l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0)

            self.cellmeasure[i] = 0.5*l
            self.glambda[i, 0, 0] = (y1 - y2)/l
            self.glambda[i, 0, 1] = (x2 - x1)/l
            self.glambda[i, 1, 0] = (y2 - y0)/l
            self.glambda[i, 1, 1] = (x0 - x2)/l
            self.glambda[i, 2, 0] = (y0 - y1)/l
            self.glambda[i, 2, 1] = (x1 - x0)/l

    @ti.kernel
    def interpolation_points(self, p: ti.u32, ipoints: ti.template()):
        """
        @brief 生成三角形网格上 p 次的插值点
        """
        NN = self.node.shape[0]
        NE = self.edge.shape[0]
        NC = self.cell.shape[0]
        GD = self.node.shape[1]
        for n in range(NN):
            for d in range(GD):
                ipoints[n, d] = self.node[n, d]
        if p > 1:
            for e in range(NE):
                s1 = NN + e*(p-1)
                for i1 in range(1, p):
                    i0 = p - i1 # (i0, i1)
                    I = s1 + i1 - 1
                    for d in range(GD):
                        ipoints[I, d] = (
                                i0*self.node[self.edge[e, 0], d] +
                                i1*self.node[self.edge[e, 1], d])/p
        if p > 2:
            cdof = (p-2)*(p-1)//2
            s0 = NN + (p-1)*NE
            for c in range(NC):
                i0 = p-2
                s1 = s0 + c*cdof
                for level in range(0, p-2):
                    i0 = p - 2 - level
                    for i2 in range(1, level+2):
                        i1 = p - i0 - i2 #(i0, i1, i2)
                        j = i1 + i2 - 2
                        I = s1 + j*(j+1)//2 + i2 - 1
                        for d in range(GD):
                            ipoints[I, d] = (
                                    i0*self.node[self.cell[c, 0], d] +
                                    i1*self.node[self.cell[c, 1], d] +
                                    i2*self.node[self.cell[c, 2], d])/p


    @ti.kernel
    def edge_to_ipoint(self, p: ti.u32, edge2ipoint: ti.template()):
        """
        @brief 返回每个边上对应 p 次插值点的全局编号
        """
        for i in range(self.edge.shape[0]):
            edge2dof[i, 0] = self.edge[i, 0]
            edge2dof[i, p] = self.edge[i, 1]
            for j in ti.static(range(1, p)):
                edge2dof[i, j] = self.node.shape[0] + i*(p-1) + j - 1

    @ti.kernel
    def cell_to_ipoint(self, p: ti.i32, cell2ipoint: ti.template()):
        """
        @brief 返回每个单元上对应 p 次插值点的全局编号
        """
        cdof = (p+1)*(p+2)//2
        NN = self.node.shape[0]
        NE = self.edge.shape[0]
        for c in range(self.cell.shape[0]):
            # 三个顶点
            cell2ipoint[c, 0] = self.cell[c, 0]
            cell2ipoint[c, cdof - p - 1] = self.cell[c, 1] # 不支持负数索引
            cell2ipoint[c, cdof - 1] = self.cell[c, 2]

            # 第 0 条边
            e = self.cell2edge[c, 0]
            v0 = self.edge[e, 0]
            s0 = NN + e*(p-1)
            s1 = cdof - p
            if v0 == self.cell[c, 1]:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + i
                    s1 += 1
            else:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + p - 2 - i
                    s1 += 1

            # 第 1 条边
            e = self.cell2edge[c, 1]
            v0 = self.edge[e, 0]
            s0 = NN + e*(p-1)
            s1 = 2
            if v0 == self.cell[c, 0]:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + i
                    s1 += i + 3
            else:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + p - 2 - i
                    s1 += i + 3

            # 第 2 条边
            e = self.cell2edge[c, 2]
            v0 = self.edge[e, 0]
            s0 = NN + e*(p-1)
            s1 = 1
            if v0 == self.cell[c, 0]:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + i
                    s1 += i + 2
            else:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + p - 2 - i
                    s1 += i + 2

            # 内部点
            if p >= 3:
                level = p - 2
                s0 = NN + (p-1)*NE + c*(p-2)*(p-1)//2
                s1 = 4
                s2 = 0
                for l in range(0, level):
                    for i in range(0, l+1):
                        cell2ipoint[c, s1] = s0 + s2
                        s1 += 1
                        s2 += 1
                    s1 += 2

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.edge.shape[0]

    def number_of_cells(self):
        return self.cell.shape[0]

    def geo_dimension(self):
        return self.node.shape[1]

    def top_dimension(self):
        return 2

    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.cell
        elif etype in {'edge', 'face', 1}:
            return self.edge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`etype` is wrong!")

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, fname, nodedata=None, celldata=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.node.to_numpy()
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1))), axis=1)

        cell = self.cell.to_numpy(dtype=np.int_)
        cellType = self.vtk_cell_type()
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV

        NC = len(cell)
        print("Writting to vtk...")
        write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                nodedata=nodedata,
                celldata=celldata)

