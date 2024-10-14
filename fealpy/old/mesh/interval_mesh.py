
from typing import Union
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh1dDataStructure


class IntervalMeshDataStructure(Mesh1dDataStructure):
    def total_face(self):
        return self.cell.reshape(-1, 1)


class IntervalMesh(Mesh, Plotable):
    ds: IntervalMeshDataStructure
    def __init__(self, node: NDArray, cell: NDArray):
        if node.ndim == 1:
            self.node = node.reshape(-1, 1)
        else:
            self.node = node

        self.ds = IntervalMeshDataStructure(len(node), cell)
        self.meshtype = 'interval'
        self.meshtype = 'INT'

        self.nodedata = {}
        self.celldata = {}
        self.edgedata = self.celldata # celldata and edgedata are the same thing
        self.facedata = self.nodedata # facedata and nodedata are the same thing

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.cell_length = self.edge_length
        self.cell_tangent = self.edge_tangent
        self.cell_unit_tangent = self.edge_unit_tangent

        self.cell_to_ipoint = self.edge_to_ipoint
        self.face_to_ipoint = self.node_to_ipoint
        self.shape_function = self._shape_function

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 0.0

    def integrator(self, q: int, etype: Union[str, int]='cell'):
        """
        @brief 返回第 k 个高斯积分公式。
        """
        from ..quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q)

    def grad_shape_function(self, bc: NDArray, p: int=1, variables: str='x', index=np.s_[:]):
        """
        @brief
        """
        R = self._grad_shape_function(bc, p=p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = np.einsum('...ij, cjm->...cim', R, Dlambda)
            return gphi
        else:
            return R

    def entity_measure(self, etype: Union[int, str]='cell', index=np.s_[:], node=None):
        """
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index, node=node)
        elif etype in {0, 'face', 'node'}:
            return np.array([0.0], dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def grad_lambda(self, index=np.s_[:]):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        v = node[cell[:, 1]] - node[cell[:, 0]]
        NC = len(cell)
        GD = self.geo_dimension()
        Dlambda = np.zeros((NC, 2, GD), dtype=self.ftype)
        h2 = np.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        Dlambda[:, 0, :] = -v
        Dlambda[:, 1, :] = v
        return Dlambda

    def prolongation_matrix(self, p0:int, p1:int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """
        assert 0 < p0 < p1

        TD = self.top_dimension()
        gdof0 = self.number_of_global_ipoints(p0)
        gdof1 = self.number_of_global_ipoints(p1)

        # 1. 网格节点上的插值点
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = np.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, TD)/p1
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.edge_shape_function(bcs[1:-1], p=p0) # (ldof1 - 2, ldof0)

        e2p1 = self.cell_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.cell_to_ipoint(p0)
        shape = (NE, ) + phi.shape

        I = np.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        return p + 1

    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    def interpolation_points(self, p: int, index=np.s_[:]) -> NDArray:
        GD = self.geo_dimension()
        node = self.entity('node')

        if p == 1:
            return node
        else:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            gdof = NN + NC*(p-1)
            ipoint = np.zeros((gdof, GD), dtype=self.ftype)
            ipoint[:NN] = node
            cell = self.entity('cell')
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            GD = self.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = np.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint

    def face_unit_normal(self, index=np.s_[:], node=None):
        """
        @brief
        """
        raise NotImplementedError

    def cell_normal(self, index=np.s_[:], node=None):
        """
        @brief 单元的法线方向
        """
        assert self.geo_dimension() == 2
        v = self.cell_tangent(index=index, node=node)
        w = np.array([(0, -1),(1, 0)])
        return v@w

    def uniform_refine(self, n=1, options={}):
        """
        @brief 一致加密网格
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
            ncell = np.zeros((2*NC, 2), dtype=self.itype)
            ncell[0:NC, 0] = cell[:, 0]
            ncell[0:NC, 1] = range(NN, NN+NC)
            ncell[NC:, 0] = range(NN, NN+NC)
            ncell[NC:, 1] = cell[:, 1]
            NN = self.node.shape[0]
            self.ds.reinit(NN, ncell)

    def refine(self, isMarkedCell, options={}):
        """
        @brief 自适应加密网格
        """

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

            self.ds.reinit(NN+N, newCell)

    ## @ingroup GeneralInterface
    def show_function(self, plot, uh, box=None):
        """
        @brief 画出定义在网格上线性有限元函数
        """
        assert self.geo_dimension() == 1
        assert self.number_of_nodes() == len(uh)

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot

        # 设置 x 轴和 y 轴的显示范围
        if box is not None:
            axes.set_xlim(box[0], box[1])
            axes.set_ylim(box[2], box[3])

        node = self.entity('node').reshape(-1)
        idx = np.argsort(node)
        line = axes.plot(node[idx], uh[idx])

         # 添加注释
        nonzero_indices = np.nonzero(uh)[0]
        for i in nonzero_indices:
            axes.text(node[i], uh[i], f'$\phi_{i}$', fontsize=12, color='b', ha='center', va='bottom')

        return line

    @classmethod
    def from_interval_domain(cls, interval=[0, 1], nx=10):
        node = np.linspace(interval[0], interval[1], nx+1, dtype=np.float64)
        cell = np.zeros((nx, 2), dtype=np.int_)
        cell[:, 0] = np.arange(0, nx)
        cell[:, 1] = np.arange(1, nx+1)
        return cls(node, cell)

    from_interval = from_interval_domain

    @classmethod
    def from_mesh_boundary(cls, mesh):
        assert mesh.top_dimension() == 2
        itype = mesh.itype
        is_bd_node = mesh.ds.boundary_node_flag()
        is_bd_face = mesh.ds.boundary_face_flag()
        node = mesh.entity('node', index=is_bd_node)
        face = mesh.entity('face', index=is_bd_face)
        NN = mesh.number_of_nodes()
        NN_bd = node.shape[0]

        I = np.zeros((NN, ), dtype=itype)
        I[is_bd_node] = np.arange(NN_bd, dtype=itype)
        face2bdnode = I[face]
        return cls(node=node, cell=face2bdnode)

    @classmethod
    def from_circle_boundary(cls, center=(0, 0), radius=1.0, n=10):
        dt = 2*np.pi/n
        theta  = np.arange(0, 2*np.pi, dt)

        node = np.zeros((n, 2), dtype=np.float64)
        cell = np.zeros((n, 2), dtype=np.int_)


        node[:, 0] = radius*np.cos(theta)
        node[:, 1] = radius*np.sin(theta)

        node[:, 0] = node[:,0] + center[0]
        node[:, 1] = node[:,1] + center[1]

        cell[:, 0] = np.arange(n)
        cell[:, 1][:-1] = np.arange(1,n)

        return cls(node, cell)

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


IntervalMesh.set_ploter('1d')
