import numpy as np


from .Mesh import Mesh

class Mesh1d(Mesh):
    def number_of_nodes(self):
        """
        @brief Get the number of nodes in the mesh.

        @return The number of nodes.
        """
        return self.NN

    def number_of_edges(self):
        """
        @brief Get the number of nodes in the mesh.

        @note `edge` is the 1D entity.

       return The number of edges.

        """
        return self.NC

    def number_of_faces(self):
        """
        @brief Get the number of nodes in the mesh.

        @note `face` is the 0D entity

        @return The number of faces.

        """
        return self.NN

    def number_of_cells(self):
        """
        @brief Get the number of cells in the mesh.

        @return The number of cells.
        """
        return self.NC

    def multi_index_matrix(self, p, etype=1):
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
        return gphi #(..., NC, ldof, GD)

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
        if GD == 2:
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
