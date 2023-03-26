import numpy as np
from types import ModuleType
from fealpy.quadrature import GaussLegendreQuadrature

## @defgroup MeshGenerators Meshgeneration algorithms on commonly used domain 
## @defgroup MeshQuality
class EdgeMesh():
    def __init__(self, node, cell):
        self.node = node
        self.itype = cell.dtype
        self.ftype = node.dtype

        self.meshtype = 'edge'
        
        self.NN = node.shape[0]
        self.GD = node.shape[1]
        self.ds = EdgeMeshDataStructure(self.NN, cell)

        self.nodedata = {}
        self.celldata = {}

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return 1

    def integrator(self, k, etype='cell'):
        """

        Notes
        -----
            返回第 k 个高斯积分公式。
        """
        return GaussLegendreQuadrature(k)

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cells(self):
        return self.ds.cell.shape[0]

    def entity(self, etype='node'):
        if etype in {'node', 'face', 0}:
            return self.node
        elif etype in {'cell', 'edge', 1}:
            return self.ds.cell
    
    def entity_measure(self, etype='cell'):
        if etype in {'cell', 'edge', 1}:
            return self.cell_length() 
        elif etype in {'node', 'face', 0}:
            return 0.0 

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

    def cell_length(self):
        node = self.entity('node')
        cell = self.entity('cell')

        v = node[cell[:, 0]] - node[cell[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return h

    def cell_unit_tangent(self):
        """
        @brief 计算每个单元的单位切向
        """
        node = self.entity('node')
        cell = self.entity('cell')
        v = node[cell[:, 0]] - node[cell[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return v/h[:, None]

    def cell_frame(self):
        """
        @brief 计算每个单元上的标架
        """
        pass


    def multi_index_matrix(self, p):
        """
        @brief 获取单元上 p 次的多重指标矩阵

        @param[in] p positive integer 

        @return multiIndex  ndarray with shape (ldof, 2)
        """
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    def shape_function(self, bc, p=1):
        """
        @brief 计算单元上的形函数在积分点 bc 处的值
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
        """
        """
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
        ldof = self.number_of_local_dofs()
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.grad_lambda()
        gphi = np.einsum('...ij, kjm->...kim', R, Dlambda[index,:,:])
        return gphi #(..., NC, ldof, GD)

    def grad_lambda(self):
        """
        @brief 重心坐标的梯度
        """
        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells()
        v = node[cell[:, 1]] - node[cell[:, 0]]
        GD = self.geo_dimension()
        Dlambda = np.zeros((NC, 2, GD), dtype=mesh.ftype)
        h2 = np.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        Dlambda[:, 0, :] = -v
        Dlambda[:, 1, :] = v
        return Dlambda

    def interpolation_points(self, p):

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
            GD = mesh.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = np.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint
    
    def add_plot(self, plot, 
            nodecolor='r',
            edgecolor='k', 
            linewidths=1, 
            aspect=None,
            markersize=10,
            box=None,
            disp=None,
            scale=1.0
            ):

        GD = self.geo_dimension()
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            if GD == 3:
                import mpl_toolkits.mplot3d as a3
                axes = fig.add_subplot(111, projection='3d')
            else:
                axes = fig.add_subplot(111)
        else:
            axes = plot

        if (aspect is None) and (GD == 3):
            axes.set_box_aspect((1, 1, 1))
            axes.set_proj_type('ortho')

        if (aspect is None) and (GD == 2):
            axes.set_box_aspect(1)


        node = self.entity('node')
        if GD == 2:
            axes.scatter(node[:, 0], node[:, 1], c=nodecolor, s=markersize)
        else:
            axes.scatter(node[:, 0], node[:, 1], node[:, 2], c=nodecolor, s=markersize)

        cell = self.entity('cell') 

        if box is None:
            if self.geo_dimension() == 2:
                box = np.zeros(4, dtype=np.float64)
            else:
                box = np.zeros(6, dtype=np.float64)

        box[0::2] = np.min(node, axis=0)
        box[1::2] = np.max(node, axis=0)

        axes.set_xlim([box[0], box[1]+0.01])
        axes.set_ylim([box[2]-0.01, box[3]])

        vts = node[cell]
        if GD == 3:
            axes.set_zlim(box[4:6])
            edges = a3.art3d.Line3DCollection(
                   vts,
                   linewidths=linewidths,
                   color=edgecolor)
            return axes.add_collection3d(edges)
        elif GD == 2:
            from matplotlib.collections import LineCollection
            edges = LineCollection(
                    vts,
                    linewidths=linewidths,
                    color=edgecolor)
            return axes.add_collection(edges)

    
    ## @ingroup MeshGenerators
    @classmethod
    def from_triangle_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tetrahedron_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tower(cls):
        node = np.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
            [-2540, -2540, 0]], dtype=np.float64)
        cell = np.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
        return cls(node, cell)



class EdgeMeshDataStructure():

    def __init__(self, NN, cell):
        self.NN = NN
        self.cell = cell 
        self.NC = len(cell)

    def cell_to_node(self):
        return self.cell

    def node_to_cell(self):
        NN = self.NN
        NC = self.NC
        I = self.cell.flat
        J = np.repeat(range(NC), 2)
        val = np.ones(2*NC, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NC))
        return node2edge
