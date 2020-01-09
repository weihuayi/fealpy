import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve

from .function import Function
from .femdof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from .femdof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d

from ..quadrature import GaussLegendreQuadrature
from ..quadrature import FEMeshIntegralAlg
"""
The following code was no use!
"""
class VectorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(
                mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.TD = self.scalarspace.TD
        self.GD = self.scalarspace.GD

    def __str__(self):
        return "Vector Lagrange finite element space!"

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def vector_dim(self):
        return self.GD

    def cell_to_dof(self):
        GD = self.GD
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = GD*cell2dof + np.arange(GD)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def boundary_dof_flag(self):
        GD = self.GD
        isBdDof = self.dof.boundary_dof()
        return np.repeat(isBdDof, GD)

    def number_of_global_dofs(self):
        return self.GD*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.GD*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bcs):
        GD = self.GD
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mn->...jmn', phi, np.eye(self.GD))
        shape += [-1, GD] 
        phi = phi.reshape(shape)
        return phi

    def div_basis(self, bcs, cellidx=None):
        gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1]
        return gphi.reshape(shape)

    def value(self, uh, bcs, cellidx=None):
        phi = self.basis(bcs)
        cell2dof = self.cell_to_dof()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum('...jm, ij->...im',  phi, uh) 
        return val 

    def div_value(self, uh, bcs, cellidx=None):
        dphi = self.div_basis(bcs, cellidx=cellidx)
        cell2dof = self.cell_to_dof()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum('...j, ij->...i',  dphi, uh) 
        return val

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=self.mesh.ftype)

    def interpolation(self, u):
        GD = self.GD
        c2d = self.dof.cell2dof
        ldof = self.dof.number_of_local_dofs()
        cell2dof = self.cell_to_dof().reshape(-1, ldof, GD)
        p = self.dof.interpolation_points()[c2d]
        uI = Function(self)
        uI[cell2dof] = u(p)
        return uI

    def stiff_matrix(self, qf, measure):
        p = self.p
        mesh = self.mesh
        GD = self.GD
        S = self.scalarspace.stiff_matrix(qf, measure)

        I, J = np.nonzero(S)
        gdof = self.number_of_global_dofs()
        A = coo_matrix(gdof, gdof)
        for i in range(self.GD):
            A += coo_matrix((S.data, (GD*I + i, GD*J + i)), shape=(gdof, gdof), dtype=mesh.ftype)
        return A.tocsr() 

    def mass_matrix(self, qf, measure, cfun=None, barycenter=True):
        p = self.p
        mesh = self.mesh
        GD = self.GD

        M = self.scalarspace.mass_matrix(qf, measure, cfun=cfun, barycenter=barycenter)
        I, J = np.nonzero(M)
        gdof = self.number_of_global_dofs()
        A = coo_matrix(gdof, gdof)
        for i in range(self.GD):
            A += coo_matrix((M.data, (GD*I + i, GD*J + i)), shape=(gdof, gdof), dtype=mesh.ftype)
        return A.tocsr() 

    def source_vector(self, f, qf, measure, surface=None):
        p = self.p
        mesh = self.mesh
        GD = self.GD
        bcs, ws = qf.quadpts, qf.weights
        pp = self.mesh.bc_to_point(bcs)
        if surface is not None:
            pp, _ = surface.project(pp)

        fval = f(pp)
        if p > 0:
            phi = self.scalarspace.basis(bcs)
            cell2dof = self.dof.cell2dof
            gdof = self.dof.number_of_global_dofs()
            b = np.zeros((gdof, GD), dtype=mesh.ftype)
            for i in range(GD):
                bb = np.einsum('i, ik, i..., k->k...', ws, fval[..., i], phi, measure)
                b[:, i]  = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        else:
            b = np.einsum('i, ikm, k->km', ws, fval,  measure)

        return b.reshape(-1)


class SymmetricTensorLagrangeFiniteElementSpace():
    #TODO: improve it 
    def __init__(self, mesh, p, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.GD = self.scalarspace.GD
        self.TD = self.scalarspace.TD

        if self.TD == 2:
            self.T = np.array([[(1, 0), (0, 0)], [(0, 1), (1, 0)], [(0, 0), (0, 1)]])
        elif self.dim == 3:
            self.T = np.array([
                [(1, 0, 0), (0, 0, 0), (0, 0, 0)], 
                [(0, 1, 0), (1, 0, 0), (0, 0, 0)],
                [(0, 0, 1), (0, 0, 0), (1, 0, 0)],
                [(0, 0, 0), (0, 1, 0), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
                [(0, 0, 0), (0, 0, 0), (0, 0, 1)]])

    def __str__(self):
        return " Symmetric Tensor Lagrange finite element space!"

    def geom_dim(self):
        return self.dim

    def tensor_dim(self):
        dim = self.dim
        return dim*(dim - 1)//2 + dim

    def cell_to_dof(self):
        tdim = self.tensor_dim()
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = tdim*cell2dof + np.arange(tdim)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def boundary_dof(self):
        tdim = self.tensor_dim()
        isBdDof = self.dof.boundary_dof()
        return np.repeat(isBdDof, tdim)

    def number_of_global_dofs(self):
        tdim = self.tensor_dim()
        return tdim*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        tdim = self.tensor_dim()
        return tdim*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bcs):
        dim = self.dim
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mno->...jmno', phi0, self.T)
        shape += [-1, dim, dim]
        return phi.reshape(shape)

    def div_basis(self, bcs, cellidx=None):
        dim = self.dim
        gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1, dim]
        return gphi.reshape(shape)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        uh0 = uh.reshape(-1, self.dim)
        if cellidx is None:
            uh0 = uh0[cell2dof].reshape(-1)
        else:
            uh0 = uh0[cell2dof[cellidx]].reshape(-1)
        val = np.einsum('...jm, ij->...im',  phi, uh0) 
        return val 

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float)

    def interpolation(self, u):
        ipoint = self.dof.interpolation_points()
        uI = Function(self)
        uI[:] = u(ipoint).flat[:]
        return uI
