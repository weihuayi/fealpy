
import numpy as np
from .vem_space import ScaledMonomialSpace2d

class VectorScaledMonomialSpace2d():
    def __init__(self, mesh, p):
        self.scalarspace = ScaledMonomialSpace2d(mesh, p)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.GD = self.scalarspace.dim # geometry dimension

    def geo_dimension(self):
        return self.GD

    def basis(self, point, cellidx=None, p=None):
        """
        Compute the basis values at point

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., M, 2) 
        """
        phi = self.scalarspace.basis(point, cellidx=cellidx, p=p)
        GD = self.GD
        phi = np.einsum('...j, mn->...jmn', phi, np.eye(GD))

        # TODO: better way?
        #shape = phi.shape[:-1] + (-1, GD)
        #phi = phi.reshape(shape)
        return phi

    def grad_basis(self, point, cellidx=None, p=None):
        GD = self.GD
        gphi0 = self.scalarspace.grad_basis(point, cellidx=cellidx, p=p)
        shape = gphi0.shape + (GD, GD)
        gphi = np.zeros(shape, dtype=np.float)
        gphi[..., 0, 0, :] = gphi0
        gphi[..., 1, 1, :] = gphi0
        return gphi

    def div_basis(self, point, cellidx=None, p=None):
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx, p=p)
        return gphi

    def grad_div_basis(self, point, cellidx=None, p=None):
        hphi = self.scalarspace.hessian_basis(point, cellidx=cellidx, p=p)
        shape = hphi.shape[:-1] + (2, 2)
        gdphi = np.zeros(shape, dtype=np.float) 
        gdphi[..., 0, 0] = hphi[..., 0]
        gdphi[..., 0, 1] = hphi[..., 2]
        gdphi[..., 1, 0] = hphi[..., 2] 
        gdphi[..., 1, 1] = hphi[..., 1]
        return gdphi

    def strain_basis(self, point, cellidx=None, p=None):
        sphi = self.scalarspace.grad_basis(point, cellidx=cellidx, p=p)
        sphi[..., 0, 0, 1] /= 2
        sphi[..., 0, 1, 0] = sphi[..., 0, 0, 1]

        sphi[..., 1, 1, 0] /= 2
        sphi[..., 1, 0, 1] = sphi[..., 1, 1, 0]
        return sphi

    def div_strain_basis(self, point, cellidx=None, p=None):
        hphi = self.scalarspace.hessian_basis(point, cellidx=cellidx, p=p)

        shape = hphi.shape[:-1] + (2, 2)
        dsphi = np.zeros(shape, dtype=np.float) 
        dsphi[..., 0, 0] = hphi[..., 0] + hphi[..., 1]/2
        dsphi[..., 0, 1] = hphi[..., 2]/2

        dsphi[..., 1, 0] = hphi[..., 2]/2
        dsphi[..., 1, 1] = hphi[..., 0]/2 + hphi[..., 1]
        return dsphi

    def value(self, uh, point, cellidx=None):
        phi = self.scalarspace(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ij->...ik', uh, phi)
        return val

    def grad_value(self, uh, point, cellidx=None):
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ijm->ikm', uh, gphi)
        return val

    def div_value(self, uh, point, cellidx=None):
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ijk->i', uh, gphi)
        return val

    def grad_div_value(self, uh, point, cellidx=None):
        hphi = self.scalarspace.hessian_basis(point, cellidx=cellidx)
        shape = hphi.shape[:-2] + (2, )
        val = np.zeros(shape, dtype=np.float)
        np.einsum('ijk, ...ijk->i', uh, hphi[..., [0, 2]], out=val[..., 0])
        np.einsum('ijk, ...ijk->i', uh, hphi[..., [2, 1]], out=val[..., 1])
        return val

    def strain_value(self, uh, point, cellidx=None):
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx)
        shape = gphi.shape[:-2] + (3, )
        val = np.zeros(shape, dtype=np.float)
        np.einsum('ijk, ...ijm->...ikm', uh, gphi, out=val[..., 0:2])
        np.einsum('ijk, ...ijk->...i', uh, gphi[..., 1::-1], out=val[..., 2])
        val[..., 2] /= 2.0
        return val

    def div_strain_value(self, uh, point, cellidx=None):
        dsphi = self.div_strain_value(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ijkm->...im', uh, dsphi)
        return val

    def function(self):
        f = FiniteElementFunction(self)
        return f 

    def array(self, dim=None):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        shape = (NC, ldof, 2)
        return np.zeros(shape, dtype=np.float)

    def number_of_local_dofs(self, p=None):
        GD = self.GD
        return GD*self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self):
        GD = self.GD
        return GD*self.dof.number_of_global_dofs()
