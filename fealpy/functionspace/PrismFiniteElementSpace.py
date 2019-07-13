import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .function import Function


class CPPFEMDof3d():

    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p

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
        face2edge =  mesh.ds.face_to_edge()

    def cell_to_dof(self):
        pass


class PrismFiniteElementSpace():

    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

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
