import numpy as np

from fealpy.quadrature import GaussLegendreQuadrature

class EdgeMesh():
    def __init__(self, node, cell):
        self.node = node
        self.itype = edge.dtype
        self.ftype = node.dtype

        self.meshtype = 'edge'
        
        self.NN = node.shape[0]
        self.ds = EdgeMeshDataStructure(self.NN, cell)

        self.nodedata = {}
        self.celldata = {}

    def geo_dimension(self):
        return self.node.shape[1]

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
    
    def entity_measure(self, etype='node'):
        if etype in {'node', 'face', 0}:
            return 0.0 
        elif etype in {'cell', 'edge', 1}:
            return self.edge_length() 

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

    def unit_cell_tangent(self):
        """
        @brief 计算每个单元的单位切向
        """
        node = self.entity('node')
        cell = self.entity('cell')
        v = node[edge[:, 0]] - node[edge[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return v/h[:, None]

    def cell_frame(self):
        """
        @brief 计算每个单元上的标架
        """
        pass

    def cell_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换到实际网格单元上的笛卡尔坐标点
        """
        node = self.node
        entity = self.entity('cell')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def multi_index_matrix(self, p, etype=2):
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
            aspect='equal',
            markersize=10,
            box=None,
            disp=None,
            scale=1.0
            ):

        GD = self.geo_dimension()
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            if GD == 3:
                from mpl_toolkits.mplot3d import Axes3D
                axes = fig.add_subplot(111, projection='3d')
            else:
                axes = fig.add_subplot(111)
        else:
            axes = plot

        axes.set_box_aspect(aspect)
        axes.set_proj_type('ortho')

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

        vts = node[edge]
        if GD == 3:
            axes.set_zlim(box[4:6])
            edges = Axes3D.art3d.Line3DCollection(
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
