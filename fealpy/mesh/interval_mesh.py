import numpy as np
from numpy.typing import NDArray
from types import ModuleType

from .mesh_base import Mesh1d, Plotable
from .mesh_data_structure import Mesh1dDataStructure, HomogeneousMeshDS


class IntervalMeshDataStructure(Mesh1dDataStructure, HomogeneousMeshDS):
    def total_face(self):
        return self.cell.reshape(-1, 1)

class IntervalMesh(Mesh1d, Plotable):
    def __init__(self, node: NDArray, cell: NDArray):
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


    def face_unit_normal(self, index=np.s_[:], node=None):
        """

        Notes
        -----
            返回点的法线向量
        """
        NN = self.number_of_nodes()
        n = np.ones(NN, dtype=self.ftype)
        n2c = self.ds.node_to_cell()
        flat = (n2c[:, 0] == n2c[:, 1]) & (n2c[:, 3] == 0)
        n[flat]= -1
        return n[index]

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
        @brief 单元的切向量 
        """
        node = self.node if node is None else node
        cell = self.entity('cell')
        v = node[cell[index, 1]] - node[cell[index, 0]]
        return v

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
    def from_interval_domain(cls, domain, nx=10):
        node = np.linspace(domain[0], domain[1], nx+1, dtype=np.float64)
        cell = np.zeros((nx, 2), dtype=np.int_)
        cell[:, 0] = np.arange(0, nx)
        cell[:, 1] = np.arange(1, nx+1)
        return cls(node, cell)

    from_interval = from_interval_domain

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

IntervalMesh.set_ploter('1d')

