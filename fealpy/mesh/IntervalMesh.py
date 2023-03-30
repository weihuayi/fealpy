import numpy as np
from .mesh_tools import unique_row, find_node, find_entity, show_mesh_1d
from scipy.sparse import csr_matrix
from types import ModuleType

from ..quadrature import GaussLegendreQuadrature

class IntervalMesh():
    def __init__(self, node, cell):
        if node.ndim == 1:
            self.node = node.reshape(-1, 1)
        else:
            self.node = node

        self.ds = IntervalMeshDataStructure(len(node), cell)
        self.meshtype = 'interval'

        self.nodedata = {}
        self.celldata = {}
        self.edgedata = self.celldata # celldata and edgedata are the same thing
        self.facedata = self.nodedata # facedata and nodedata are the same thing 

        self.itype = cell.dtype
        self.ftype = node.dtype


    def integrator(self, k, etype='cell'):
        return GaussLegendreQuadrature(k)

    def geo_dimension(self):
        return self.node.shape[-1]

    def top_dimension(self):
        return 1

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_faces(self):
        return self.ds.NN

    def number_of_edges(self):
        return self.ds.NC

    def number_of_cells(self):
        return self.ds.NC

    def number_of_entities(self, etype):
        if etype in {'cell', 'edge', 1}:
            return self.ds.NC
        elif etype in {'node', 'face', 0}:
            return self.ds.NN
        else:
            raise ValueError(f"etype {etype} must be 0 or 1!") #TODO

    def bc_to_point(self, bc, index=np.s_[:], node=None):
        """

        Notes
        -----
            把重心坐标转换为实际空间坐标
        """
        node = self.node if node is None else node
        cell = self.entity('cell')
        p = np.einsum('...j, ijk->...ik', bc, node[cell[index]])
        return p

    def entity(self, etype=1):
        if etype in {'cell', 'edge', 1}:
            return self.ds.cell
        elif etype in {'node', 'face', 0}:
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")#TODO

    def entity_measure(self, etype=1, index=np.s_[:], node=None):
        """
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index, node=None)
        elif etype in {0, 'face', 'node'}:
            return 0
        else:
            raise ValueError("`etype` is wrong!")

    def entity_barycenter(self, etype=1, index=np.s_[:], node=None):
        """

        Notes
        -----
            返回网格实体的重心坐标。

            注意，这里用户可以提供一个新个网格节点数组。
        """
        node = self.entity('node') if node is None else node
        if etype in {1, 'cell',  'edge'}:
            cell = self.ds.cell
            bc = np.sum(node[cell[index]], axis=1)/cell.shape[-1]
        elif etype in {'node', 'face', 0}:
            bc = node[index]
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity)) 
        return bc

    def multi_index_matrix(self, p):
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    def shape_function(self, bc, p=1):
        """
        @brief 
        """
        TD = bc.shape[-1] - 1 
        multiIndex = self.multi_index_matrix(p)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def grad_shape_function(self, bc, p=1, index=np.s_[:]):

        TD = self.top_dimension()

        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_ipoints(p)
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.grad_lambda(index=index)
        gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
        return gphi #(..., NC, ldof, GD)

    def grad_lambda(self, index=np.s_[:]):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells() if index == np.s_[:] else len(index)
        v = node[cell[index, 1]] - node[cell[index, 0]]
        GD = self.geo_dimension()
        Dlambda = np.zeros((NC, 2, GD), dtype=self.ftype)
        h2 = np.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        Dlambda[:, 0, :] = -v
        Dlambda[:, 1, :] = v
        return Dlambda

    def number_of_local_ipoints(self, p, iptype='cell'):
        if iptype in {'cell', 'edge', 1}:
            return p+1 
        elif iptype in {'node', 'face', 0}:
            return 1
    
    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC
    
    def interpolation_points(self, p):
        """
        @brief 获取三角形网格上所有 p 次插值点
        """
        cell = self.entity('cell')
        node = self.entity('node')
        if p == 1:
            return node


    def vtk_cell_type(self):
        VTK_LINE = 3
        return VTK_LINE

    def to_vtk(self, etype='edge', index=np.s_[:], fname=None):
        """

        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD < 3:
            node = np.c_[node, np.zeros((node.shape[0], 3-GD))]

        cell = self.entity(etype)[index]
        NV = cell.shape[-1]
        NC = len(cell)

        cell = np.c_[np.zeros((NC, 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV

        cellType = self.vtk_cell_type()  # segment 

        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)



    def cell_length(self, index=np.s_[:], node=None):
        """

        Notes
        -----
            返回单元的长度。
        """
        node = self.node if node is None else node
        cell = self.ds.cell
        GD = self.geo_dimension()
        return np.sqrt(np.sum((node[cell[index, 1]] - node[cell[index, 0]])**2,
                        axis=-1))


    def cell_normal(self, index=np.s_[:], node=None):
        """

        Notes
        -----
            返回二维空间中单元的法线向量
        """
        GD = self.geo_dimension()
        if GD != 2:
            raise ValueError('cell_normal just work for 2D Case')
        v = self.cell_tangent(index=index, node=node)
        w = np.array([(0, -1),(1, 0)])
        return v@w

    def cell_tangent(self, index=np.s_[:], node=None):
        """

        Notes
        -----
            返回单元的切向向量
        """
        node = self.node if node is None else node
        cell = self.entity('cell')
        v = node[cell[index, 1]] - node[cell[index, 0]]
        return v

    def uniform_refine(self, n=1, inplace=True):
        """

        Notes
        -----
            对网格进行一致加密，

            inplace 默认为 True， 意思是直接在单元内部修改

        TODO:
            1. 实现 inplace 为 False 的情形 
        """
        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            node = self.entity('node')
            cell = self.entity('cell')
            cell2newNode = np.arange(NN, NN+NC)
            newNode = (node[cell[:, 0]] + node[cell[:, 1]])/2
            self.node = np.r_['0', node, newNode]
            p = np.r_['-1', cell, cell2newNode.reshape(-1,1)]
            ncell = np.zeros((2*NC, 2), dtype=np.int)
            ncell[0:NC, 0] = cell[:, 0]
            ncell[0:NC, 1] = range(NN, NN+NC)
            ncell[NC:, 0] = range(NN, NN+NC)
            ncell[NC:, 1] = cell[:, 1]
            NN = self.node.shape[0]
            self.ds.reinit(NN, ncell)

    def refine(self, isMarkedCell, inplace=True):

        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells()
        NN = self.number_of_nodes()

        N = isMarkedCell.sum()
        if N > 0:
            bc = self.entity_barycenter('cell', index=isMarkedCell)
            self.node = np.r_['0', node, bc] #将新的节点添加到总的节点中去，得到的node

            newCell = np.zeros((NC+N, 2), dtype=self.itype)
            newCell[:NC] = cell
            newCell[:NC][isMarkedCell, 1] = range(NN, NN+N)
            newCell[NC:, 0] = range(NN, NN+N)
            newCell[NC:, 1] = cell[isMarkedCell, 1]
            print(newCell)
            print(self.node)

            self.ds.reinit(NN+N, newCell)


    def add_plot(self, plot,
            nodecolor='k', cellcolor='k',
            aspect='equal', linewidths=1, markersize=20,
            showaxis=False):

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        return show_mesh_1d(axes, self,
                nodecolor=nodecolor, cellcolor=cellcolor, aspect=aspect,
                linewidths=linewidths, markersize=markersize,
                showaxis=showaxis)

    def find_node(self, axes, node=None,
            index=None, showindex=False,
            color='r', markersize=200,
            fontsize=24, fontcolor='k'):

        if node is None:
            node = self.node
        find_node(axes, node,
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_edge(self, axes,
            index=None, showindex=False,
            color='g', markersize=250,
            fontsize=24, fontcolor='g'):

        find_entity(axes, self, entity='edge',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_face(self, axes,
            index=None, showindex=False,
            color='g', markersize=250,
            fontsize=24, fontcolor='g'):

        find_entity(axes, self, entity='edge',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_cell(self, axes,
            index=None, showindex=False,
            color='g', markersize=250,
            fontsize=24, fontcolor='g'):

        find_entity(axes, self, entity='cell',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

class IntervalMeshDataStructure():
    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = len(cell)
        self.cell = cell
        self.itype = cell.dtype
        self.construct()

    def reinit(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def construct(self):
        NN = self.NN
        NC = self.NC
        cell = self.cell

        _, i0, j = np.unique(cell.reshape(-1), 
                return_index=True, return_inverse=True)
        self.node2cell = np.zeros((NN, 4), dtype=self.itype)

        i1 = np.zeros(NN, dtype=np.int) 
        i1[j] = np.arange(2*NC)

        self.node2cell[:, 0] = i0//2 
        self.node2cell[:, 1] = i1//2
        self.node2cell[:, 2] = i0%2 
        self.node2cell[:, 3] = i1%2 

    def cell_to_node(self):
        return self.cell

    def cell_to_cell(self):
        NC = self.NC
        node2cell = self.node2cell
        cell2cell = np.zeros((NC, 2), dtype=np.int)
        cell2cell[node2cell[:, 0], node2cell[:, 2]] = node2cell[:, 1]
        cell2cell[node2cell[:, 1], node2cell[:, 3]] = node2cell[:, 0]
        return cell2cell

    def node_to_cell(self):
        return self.node2cell

    def node_to_node(self):
        NN = self.NN
        NC = self.NC
        cell = self.cell
        I = cell.flatten()
        J = cell[:,[1,0]].flatten()
        val = np.ones((2*NC,), dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN),dtype=np.bool_)
        return node2node

    def boundary_node_flag(self):
        node2cell = self.node2cell
        return node2cell[:, 0] == node2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.NC
        node2cell = self.node2cell
        isBdCell = np.zeros((NC,), dtype=np.bool_)
        isBdNode = self.boundary_node_flag()
        isBdCell[node2cell[isBdNode, 0]] = True
        return isBdCell 

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx 

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx 
