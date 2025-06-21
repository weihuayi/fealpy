import numpy as np

from types import ModuleType
from .Mesh import Mesh
from ...quadrature import GaussLegendreQuadrature
from ...quadrature import ZeroDimensionQuadrature

## @defgroup GeneralInterface
class Mesh1d(Mesh):

    def top_dimension(self):
        """
        @brief
        """
        return 1

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

    def integrator(self, k, etype='cell'):
        """

        Notes
        -----
            返回第 k 个高斯积分公式。
        """
        if etype in {'cell', 'edge', 1}:
            return GaussLegendreQuadrature(k)
        elif etype in {'node', 'face', 0}:
            return ZeroDimensionQuadrature(k) 

    def number_of_local_ipoints(self, p, iptype='cell'):
        return p+1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    def interpolation_points(self, p, index=np.s_[:]):
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

    def node_to_ipoint(self, p, index=np.s_[:]):
        return np.arange(self.number_of_nodes())[:, None]

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格边与插值点的对应关系
        """
        NC = self.number_of_cells()
        NN = self.number_of_nodes()

        cell = self.entity('cell')
        cell2ipoints = np.zeros((NC, p+1), dtype=np.int_)
        cell2ipoints[:, [0, -1]] = cell
        if p > 1:
            cell2ipoints[:, 1:-1] = NN + np.arange(NC*(p-1)).reshape(NC, p-1)
        return cell2ipoints[index]

    edge_to_ipoint = cell_to_ipoint
    face_to_ipoint = node_to_ipoint

    def multi_index_matrix(self, p, etype=1):
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    def shape_function(self, bc, p=1, etype='cell'):
        """
        @brief 
        """
        if etype in {'cell', 'edge', 1}:

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
        elif etype in {'face', 'node', 0}:
            phi = np.array([[1]], dtype=self.ftype)
        return phi


    ## @ingroup FEMInterface
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
        return gphi 

    def add_plot(self, plot, nodecolor='k', cellcolor='k', aspect='equal', 
                linewidths=1, markersize=20, showaxis=False):
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot

        axes.set_aspect(aspect)
        if showaxis == False:
            axes.set_axis_off()
        else:
            axes.set_axis_on()

        node = self.entity('node')

        if len(node.shape) == 1:
            node = node[:, None]

        if node.shape[1] == 1:
            node = np.r_['1', node, np.zeros_like(node)]

        GD = self.geo_dimension()
        if GD == 1:
            axes.scatter(node[:, 0], node[:, 1], color=nodecolor, s=markersize)
        elif GD == 2:
            axes.scatter(node[:, 0], node[:, 1], color=nodecolor, s=markersize)
        elif GD == 3:
            axes.scatter(node[:, 0], node[:, 1], node[:, 2], color=nodecolor, s=markersize)

        cell = self.entity('cell')
        vts = node[cell, :]

        if GD < 3:
            from matplotlib.collections import LineCollection
            lines = LineCollection(vts, linewidths=linewidths, colors=cellcolor)
            return axes.add_collection(lines)
        else:
            import mpl_toolkits.mplot3d as a3
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            lines = Line3DCollection(vts, linewidths=linewidths, colors=cellcolor)
            return axes.add_collection3d(vts)


