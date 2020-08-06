import numpy as np
from ..quadrature import TriangleQuadrature

from .Mesh2d import Mesh2d, Mesh2dDataStructure

from .TriangleMesh import TriangleMesh

from .multi_index import multi_index_matrix1d
from .multi_index import multi_index_matrix2d
from .multi_index import multi_index_matrix3d

class LagrangeTriangleMesh(Mesh2d):
    def __init__(self, node, cell, p=1, surface=None):

        mesh = TriangleMesh(node, cell) 
        dof = LagrangeTriangleDof2d(mesh, p)

        self.p = p
        self.node = dof.interpolation_points()

        if surface is not None:
            self.node, _ = surface.project(self.node)
   
        self.ds = LagrangeTriangleMeshDataStructure(dof)

        self.TD = 2
        self.GD = node.shape[1]

        self.meshtype = 'ltri'
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}

        self.multi_index_matrix = [multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]

    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
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
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        index = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, index]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def cell_area(self, index=None):
        pass

    def bc_to_point(self, bc, etype='cell', index=np.s_[:]):
        node = self.node
        entity = self.entity(etype) # default  cell
        phi = self.lagrange_basis(bc, etype=etype) # (NQ, 1, ldof)
        p = np.einsum('ijk, jkn->ijn', phi, node[entity[index]])
        return p

    def lagrange_basis(self, bc, etype='cell'):
        p = self.p   # the degree of lagrange basis function

        if etype in {'cell', 2}:
            TD = 2
        elif etype in {'edge', 'face', 1}:
            TD = 1

        multiIndex = self.multi_index_matrix[TD-1](p)

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi[..., np.newaxis, :] # (..., 1, ldof)

    def jacobi_matrix(self, bc, index=np.s_[:]):
        mesh = self.mesh
        cell = mesh.entity('cell')
        grad = self.grad_basis(bc, index=index)

        # the tranpose of the jacobi matrix between S_h and K
        NV = self.ds.number_of_nodes_of_cells()
        J = node[cell[index, [NV-1-p, NV-1]]] - node[cell[index, [0]]]
        Jh = mesh.jacobi_matrix(index=index)

        # the tranpose of the jacobi matrix between S_p and S_h
        Jph = np.einsum(
                'ijm, ...ijk->...imk',
                self.node[cell2dof[index], :],
                grad)

        # the transpose of the jacobi matrix between S_p and K
        Jp = np.einsum('...ijk, imk->...imj', Jph, Jh)
        grad = np.einsum('ijk, ...imk->...imj', Jh, grad)
        return Jp, grad

    def grad_basis(self, bc, index=np.s_[:]):

        p = self.p   # the degree of polynomial basis function
        TD = self.TD

        multiIndex = self.multi_index_matrix[TD](p) 

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

        # Dlambda = self.mesh.grad_lambda()
        gphi = np.einsum('...ij, kjm->...kim', R, Dlambda[index, :, :])
        return gphi #(..., NC, ldof, GD)

    def print(self):

        """

        Notes
        -----
            打印网格信息， 用于调试。
        """

        node = self.entity('node')
        print('node:')
        for i, v in enumerate(node):
            print(i, ": ", v)

        edge = self.entity('edge')
        print('edge:')
        for i, e in enumerate(edge):
            print(i, ": ", e)

        cell = self.entity('cell')
        print('cell:')
        for i, c in enumerate(cell):
            print(i, ": ", c)

        edge2cell = self.ds.edge_to_cell()
        print('edge2cell:')
        for i, ec in enumerate(edge2cell):
            print(i, ": ", ec)



class LagrangeTriangleMeshDataStructure(Mesh2dDataStructure):
    def __init__(self, dof):
        self.cell = dof.cell_to_dof()
        self.edge = dof.edge_to_dof()
        self.edge2cell = dof.mesh.ds.edge_to_cell()

        self.NN = dof.number_of_global_dofs() 
        self.NE = len(self.edge)
        self.NC = len(self.cell)

        self.V = dof.number_of_local_dofs() 
        self.E = 3

class LagrangeTriangleDof2d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix2d(p)

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) > 0
        return isNodeDof

    def is_on_edge_local_dof(self):
        return self.multiIndex == 0

    def is_boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        return self.edge_to_dof()

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE= mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, -1]] = edge
        if p > 1:
            edge2dof[:, 1:-1] = NN + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh

        cell = mesh.entity('cell')
        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        if p == 1:
            cell2dof = cell

        if p > 1:
            cell2dof = np.zeros((NC, ldof), dtype=np.int)

            isEdgeDof = self.is_on_edge_local_dof()
            edge2dof = self.edge_to_dof()
            cell2edgeSign = mesh.ds.cell_to_edge_sign()
            cell2edge = mesh.ds.cell_to_edge()

            cell2dof[np.ix_(cell2edgeSign[:, 0], isEdgeDof[:, 0])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 0], [0]], :]
            cell2dof[np.ix_(~cell2edgeSign[:, 0], isEdgeDof[:,0])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 0], [0]], -1::-1]

            cell2dof[np.ix_(cell2edgeSign[:, 1], isEdgeDof[:, 1])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 1], [1]], -1::-1]
            cell2dof[np.ix_(~cell2edgeSign[:, 1], isEdgeDof[:,1])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 1], [1]], :]

            cell2dof[np.ix_(cell2edgeSign[:, 2], isEdgeDof[:, 2])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 2], [2]], :]
            cell2dof[np.ix_(~cell2edgeSign[:, 2], isEdgeDof[:,2])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 2], [2]], -1::-1]
        if p > 2:
            base = N + (p-1)*NE
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            idof = ldof - 3*p
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == 1:
            return node
        if p > 1:
            N = node.shape[0]
            dim = node.shape[-1]
            gdof = self.number_of_global_dofs()
            ipoint = np.zeros((gdof, dim), dtype=np.float)
            ipoint[:N, :] = node
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1,2), dtype=np.float)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                    node[edge,:]).reshape(-1, dim)
        if p > 2:
            isEdgeDof = self.is_on_edge_local_dof()
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            w = self.multiIndex[isInCellDof, :]/p
            ipoint[N+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell,:]).reshape(-1, dim)

        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        N = self.mesh.number_of_nodes()
        gdof = N
        if p > 1:
            NE = self.mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            ldof = self.number_of_local_dofs()
            NC = self.mesh.number_of_cells()
            gdof += (ldof - 3*p)*NC
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)//2 
        elif doftype in {'face', 'edge',  1}:
            return self.p + 1
        elif doftype in {'node', 0}:
            return 1
