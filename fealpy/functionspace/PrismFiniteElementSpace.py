import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .function import Function
from timeit import default_timer as timer


class CPPFEMDof3d():

    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.cell2dof = self.cell_to_dof()
        self.dpoints = self.interpolation_points()

    def multi_index_matrix(self):
        p = self.p
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int8)
        multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 1] = idx0 - multiIndex[:, 2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex

    def number_of_local_dofs(self):
        p = self.p
        return (p+1)*(p+1)*(p+2)//2

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NTF = mesh.number_of_tri_faces()
        NQF = mesh.number_of_quad_faces()
        NC = mesh.number_of_cells()
        gdof = NN

        if p > 1:
            gdof += NE*(p-1) + NQF*(p-1)*(p-1)

        if p > 2:
            tfdof = (p+1)*(p+2)//2 - 3*p
            gdof += NTF*tfdof
            gdof += NC*tfdof*(p-1)

        return gdof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, -1]] = edge

        if p > 1:
            edge2dof[:, 1:-1] = NN + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def face_to_dof(self):
        p = self.p
        mesh = self.mesh
        face2edge = mesh.ds.face_to_edge()

    def cell_to_dof(self):
        start = timer()
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()

        cell = mesh.entity('cell') + p*p + 1

        ldof = self.number_of_local_dofs()
        w1 = np.zeros((p+1, 2), dtype=np.int8)
        w1[:, 0] = np.arange(p, -1, -1)
        w1[:, 1] = w1[-1::-1, 0]
        w2 = self.multi_index_matrix()
        w3 = np.einsum('ij, km->ijkm', w1, w2)

        w = np.zeros((ldof, 6), dtype=np.int8)
        w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
        w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)

        dtype = np.dtype([('nidx', np.int), ('widx', np.int8)])
        ps = np.zeros((NC, ldof, 6), dtype=dtype)
        ps['nidx'][:, :, :] = cell[:, np.newaxis, :]
        ps['widx'][:, :, :] = w[np.newaxis, :, :]
        ps['nidx'][ps['widx'] == 0] = -1
        ps.sort()
        t, self.i0, j = np.unique(
                ps.reshape(-1, 6),
                return_index=True,
                return_inverse=True,
                axis=0)
        end = timer()
        print('time is:', end - start)
        return j.reshape(-1, ldof)

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        GD = mesh.geo_dimension()

        ldof = self.number_of_local_dofs()
        w1 = np.zeros((p+1, 2), dtype=np.float)
        w1[:, 0] = np.arange(p, -1, -1)/p
        w1[:, 1] = w1[-1::-1, 0]
        w2 = self.multi_index_matrix()/p
        w3 = np.einsum('ij, km->ijkm', w1, w2)
        w = np.zeros((ldof, 6), dtype=np.float)
        w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
        w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)
        ps = np.einsum('km, imd->ikd', w, node[cell]).reshape(-1, GD)
        ipoint = ps[self.i0]

        return ipoint


class PrismFiniteElementSpace():

    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.dpoints

    def cell_to_dof(self):
        return self.dof.cell2dof

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def basis(self, bc):
        pass

    def grad_basis(self, bc, cellidx=None):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)
