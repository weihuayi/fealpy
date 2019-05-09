import numpy as np
from .function import Function
from ..quadrature import GaussLobattoQuadrature


class SMDof2d():
    """
    单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = self.multi_index_matrix()
        self.cell2dof = self.cell_to_dof()

    def multi_index_matrix(self):
        """
        Compute the natural correspondence from the one-dimensional index
        starting from 0.

        Notes
        -----

        0<-->(0, 0), 1<-->(1, 0), 2<-->(0, 1), 3<-->(2, 0), 4<-->(1, 1),
        5<-->(0, 2), .....

        """
        ldof = self.number_of_local_dofs()
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex

    def cell_to_dof(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self, p=None):
        if p is None:
            p = self.p
        return (p+1)*(p+2)//2

    def number_of_global_dofs(self):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        return NC*ldof


class ScaledMonomialSpace2d():
    def __init__(self, mesh, p):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.barycenter = mesh.entity_barycenter('cell')
        self.p = p
        self.area = mesh.entity_measure('cell')
        self.h = np.sqrt(self.area)
        self.dof = SMDof2d(mesh, p)
        self.GD = 2

    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self):
        return self.dof.cell2dof

    def basis(self, point, cellidx=None, p=None):
        """
        Compute the basis values at point

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., M, 2), M is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., M, ldof)

        """
        if p is None:
            p = self.p
        h = self.h
        NC = self.mesh.number_of_cells()

        ldof = self.number_of_local_dofs(p=p)
        if p == 0:
            shape = point.shape[:-1] + (1, )
            return np.ones(shape, dtype=np.float)

        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float)  # (..., M, ldof)
        if cellidx is None:
            assert(point.shape[-2] == NC)
            phi[..., 1:3] = (point - self.barycenter)/h.reshape(-1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            phi[..., 1:3] = (point - self.barycenter[cellidx])/h[cellidx].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1

        return phi

    def value(self, uh, point, cellidx=None):
        phi = self.basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            return np.einsum('ij, ...ij->...i', uh[cell2dof], phi)
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ij->...i', uh[cell2dof[cellidx]], phi)

    def grad_basis(self, point, cellidx=None, p=None):
        if p is None:
            p = self.p
        h = self.h
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 2)
        gphi = np.zeros(shape, dtype=np.float)
        gphi[..., 1, 0] = 1
        gphi[..., 2, 1] = 1
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            phi = self.basis(point, cellidx=cellidx)
            for i in range(2, p+1):
                gphi[..., start:start+i, 0] = np.einsum('i, ...i->...i', r[i-1::-1], phi[..., start-i:start])
                gphi[..., start+1:start+i+1, 1] = np.einsum('i, ...i->...i', r[0:i], phi[..., start-i:start])
                start += i+1
        if cellidx is None:
            return gphi/h.reshape(-1, 1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            return gphi/h[cellidx].reshape(-1, 1, 1)

    def grad_value(self, uh, point, cellidx=None):
        gphi = self.grad_basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            return np.einsum('ij, ...ijm->...im', uh[cell2dof], gphi)
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ijm->...im', uh[cell2dof[cellidx]], gphi)

    def laplace_basis(self, point, cellidx=None, p=None):
        if p is None:
            p = self.p
        area = self.area

        ldof = self.number_of_local_dofs() 

        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:] 
            phi = self.basis(point, cellidx=cellidx)
            for i in range(2, p+1):
                lphi[..., start:start+i-1] += np.einsum('i, ...i->...i', r[i-2::-1], phi[..., start-2*i+1:start-i])
                lphi[..., start+2:start+i+1] += np.eisum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                start += i+1

        if cellidx is None:
            return lphi/area.reshape(-1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            return lphi/area[cellidx].reshape(-1, 1)

    def hessian_basis(self, point, cellidx=None, p=None):
        """
        Compute the value of the hessian of the basis at a set of 'point'

        Parameters
        ----------
        point : numpy array
            The shape of point is (..., NC, 2)

        Returns
        -------
        hphi : numpy array
            the shape of hphi is (..., NC, ldof, 3)
        """
        if p is None:
            p = self.p

        area = self.area
        ldof = self.number_of_local_dofs()

        shape = point.shape[:-1]+(ldof, 3)
        hphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:]
            phi = self.basis(point, cellidx=cellidx)
            for i in range(2, p+1):
                hphi[..., start:start+i-1, 0] = np.einsum('i, ...i->...i', r[i-2::-1], phi[..., start-2*i+1:start-i])
                hphi[..., start+2:start+i+1, 1] = np.einsum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                r0 = np.arange(1, i)
                r0 = r0*r0[-1::-1]
                hphi[..., start+1:start+i, 2] = np.einsum('i, ...i->...i', r0, phi[..., start-2*i+1:start-i])
                start += i+1

        if cellidx is None:
            return hphi/area.reshape(-1, 1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            return hphi/area[cellidx].reshape(-1, 1, 1)

    def laplace_value(self, uh, point, cellidx=None):
        lphi = self.laplace_basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            return np.einsum('ij, ...ij->...i', uh[cell2dof], lphi)
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ij->...i', uh[cell2dof[cellidx]], lphi)

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
        return np.zeros(shape, dtype=np.float)

    def number_of_local_dofs(self, p=None):
        return self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()
