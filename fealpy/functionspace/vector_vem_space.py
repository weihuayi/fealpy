
import numpy as np
from .vem_space import ScaledMonomialSpace2d, VEMDof2d
from .function import Function

class VectorScaledMonomialSpace2d():
    def __init__(self, mesh, p):
        self.scalarspace = ScaledMonomialSpace2d(mesh, p)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.GD = self.scalarspace.GD # geometry dimension
        self.area = self.scalarspace.area
        self.barycenter = self.scalarspace.barycenter

    def geo_dimension(self):
        return self.GD

    def basis(self, point, cellidx=None, p=None):
        """
        Compute the basis values at point

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., NC, 2) 

        Returns
        -------
        phi : numpy array
            the shape of `phi` is (..., NC, ldof*GD, GD)
        """
        phi = self.scalarspace.basis(point, cellidx=cellidx, p=p)
        phi = np.einsum('...j, mn->...jmn', phi, np.eye(self.GD))

        # TODO: better way?
        shape = phi.shape[:-3] + (-1, self.GD)
        return phi.reshape(shape)

    def grad_basis(self, point, cellidx=None, p=None):
        """
        Compute the value of the gradients of basis at a set of point

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., NC, 2) 

        Returns
        -------
        gphi : numpy array
            the shape of gphi is (..., NC, ldof*GD, GD, GD)
        """
        GD = self.GD
        gphi0 = self.scalarspace.grad_basis(point, cellidx=cellidx, p=p)
        shape = gphi0.shape + (GD, GD)
        gphi = np.zeros(shape, dtype=np.float)
        gphi[..., 0, 0, :] = gphi0
        gphi[..., 1, 1, :] = gphi0

        shape = gphi.shape[:-4] + (-1, GD, GD)
        return  gphi.reshape(shape)

    def div_basis(self, point, cellidx=None, p=None):
        """
        Compute the value of the divergence of the basis at a set of 'point'

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., NC, 2) 

        Returns
        -------
        dphi : numpy array
            the shape of gphi is (..., NC, ldof*GD)
        """
        dphi = self.scalarspace.grad_basis(point, cellidx=cellidx, p=p)

        shape = dphi.shape[:-2] + (-1, )
        return dphi.reshape(shape)

    def grad_div_basis(self, point, cellidx=None, p=None):
        """
        Compute the value of the gradient of the divergence of the basis at a set of 'point'

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., NC, 2) 

        Returns
        -------
        gdphi : numpy array
            the shape of gdphi is (..., NC, ldof*GD, GD)
        """
        hphi = self.scalarspace.hessian_basis(point, cellidx=cellidx, p=p)
        GD = self.GD
        shape = hphi.shape[:-1] + (GD, GD)
        gdphi = np.zeros(shape, dtype=np.float) 
        gdphi[..., 0, 0] = hphi[..., 0]
        gdphi[..., 0, 1] = hphi[..., 2]
        gdphi[..., 1, 0] = hphi[..., 2] 
        gdphi[..., 1, 1] = hphi[..., 1]
        shape = gdphi.shape[:-3] + (-1, GD)
        return gdphi.reshape(shape)

    def strain_basis(self, point, cellidx=None, p=None):
        """
        Compute the value of  the strain of the basis at a set of 'point'

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., NC, 2) 

        Returns
        -------
        sdphi : numpy array
            the shape of sphi is (..., NC, ldof*GD, GD, GD)
        """
        GD = self.GD
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx, p=p)
        shape = gphi.shape + (GD, GD)
        sphi = np.zeros(shape, dtype=np.float)
        
        sphi[..., 0, 0, 0] = gphi[..., 0]
        sphi[..., 0, 0, 1] = gphi[..., 1]/2
        sphi[..., 0, 1, 0] = sphi[..., 0, 0, 1]

        sphi[..., 1, 1, 1] = gphi[..., 1]
        sphi[..., 1, 1, 0] = gphi[..., 0]/2
        sphi[..., 1, 0, 1] = sphi[..., 1, 1, 0]

        shape = sphi.shape[:-4] + (-1, GD, GD)
        return sphi.reshape(shape)

    def div_strain_basis(self, point, cellidx=None, p=None):
        """
        Compute the value of  the divergence of the strain of the basis at a set of 'point'

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., NC, 2) 

        Returns
        -------
        sdphi : numpy array
            the shape of sphi is (..., NC, ldof*GD, GD)
        """
        hphi = self.scalarspace.hessian_basis(point, cellidx=cellidx, p=p)

        shape = hphi.shape[:-1] + (2, 2)
        dsphi = np.zeros(shape, dtype=np.float) 
        dsphi[..., 0, 0] = hphi[..., 0] + hphi[..., 1]/2
        dsphi[..., 0, 1] = hphi[..., 2]/2

        dsphi[..., 1, 0] = hphi[..., 2]/2
        dsphi[..., 1, 1] = hphi[..., 0]/2 + hphi[..., 1]
        shape = dsphi.shape[:-3] + (-1, self.GD)
        return dsphi.reshape(shape)

    def value(self, uh, point, cellidx=None):
        """
        Compute the value of the scaled monomial function

        Parameters
        ---------- 
        uh : numpy array
            The shape of uh is (NC, ldof, GD)
        point : numpy array
            The shape of point is (..., NC, GD) 

        Returns
        -------
        val : numpy array
            the shape of val is (..., NC, GD)
        """
        phi = self.scalarspace(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ij->...ik', uh, phi)
        return val

    def grad_value(self, uh, point, cellidx=None):
        """
        Compute the gradient value of the scaled monomial function

        Parameters
        ---------- 
        uh : numpy array
            The shape of uh is (NC, ldof, GD)
        point : numpy array
            The shape of point is (..., NC, GD) 

        Returns
        -------
        val : numpy array
            the shape of val is (..., NC, GD, GD)
        """
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ijm->...ikm', uh, gphi)
        return val

    def div_value(self, uh, point, cellidx=None):
        """
        Compute the divergence value of the scaled monomial function

        Parameters
        ---------- 
        uh : numpy array
            The shape of uh is (NC, ldof, GD)
        point : numpy array
            The shape of point is (..., NC, GD) 

        Returns
        -------
        val : numpy array
            the shape of val is (..., NC)
        """
        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx)
        val = np.einsum('ijk, ...ijk->...i', uh, gphi)
        return val

    def grad_div_value(self, uh, point, cellidx=None):
        """
        Compute the gradient value of the divergence of the scaled monomial
        function `uh`.

        Parameters
        ---------- 
        uh : numpy array
            The shape of uh is (NC, ldof, GD)
        point : numpy array
            The shape of point is (..., NC, GD) 

        Returns
        -------
        val : numpy array
            the shape of val is (..., NC, GD)
        """
        hphi = self.scalarspace.hessian_basis(point, cellidx=cellidx)
        shape = hphi.shape[:-2] + (self.GD, )
        val = np.zeros(shape, dtype=np.float)
        np.einsum('ijk, ...ijk->...i', uh, hphi[..., [0, 2]], out=val[..., 0])
        np.einsum('ijk, ...ijk->...i', uh, hphi[..., [2, 1]], out=val[..., 1])
        return val

    def strain_value(self, uh, point, cellidx=None):
        """
        Compute the strain value of the scaled monomial
        function `uh`.

        Parameters
        ---------- 
        uh : numpy array
            The shape of uh is (NC, ldof, GD)
        point : numpy array
            The shape of point is (..., NC, GD) 

        Returns
        -------
        val : numpy array
            the shape of val is (..., NC, 3)
        """

        gphi = self.scalarspace.grad_basis(point, cellidx=cellidx)
        shape = gphi.shape[:-2] + (3, )
        val = np.zeros(shape, dtype=np.float)
        np.einsum('ijk, ...ijm->...ikm', uh, gphi, out=val[..., 0:2])
        np.einsum('ijk, ...ijk->...i', uh, gphi[..., 1::-1], out=val[..., 2])
        val[..., 2] /= 2.0
        return val

    def div_strain_value(self, uh, point, cellidx=None):
        """
        Compute the divergence value of  the scaled monomial
        function `uh`.

        Parameters
        ---------- 
        uh : numpy array
            The shape of uh is (NC, ldof, GD)
        point : numpy array
            The shape of point is (..., NC, GD) 

        Returns
        -------
        val : numpy array
            the shape of val is (..., NC, GD)
        """
        dsphi = self.div_strain_value(point, cellidx=cellidx)
        shape = uh.shape[:-2] + (-1, )
        val = np.einsum('ij, ...ijm->...im', uh.reshape(shape), dsphi)
        return val

    def function(self):
        f = Function(self)
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

class VectorVirtualElementSpace2d():
    def __init__(self, mesh, p = 1):
        self.mesh = mesh
        self.p = p
        self.vsmspace = VectorScaledMonomialSpace2d(mesh, p)
        self.dof = VEMDof2d(mesh, p) # 标量空间自由度管理对象
        self.GD = 2

    def cell_to_dof(self):
        
        NC = self.mesh.number_of_cells()
        GD = self.GD
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = (GD*cell2dof + np.arange(GD)).reshape(-1)
        NDof = GD*self.dof.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC+1, dtype=self.mesh.itype)
        cell2dofLocation[1:] = np.add.accumulate(NDof)
        
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        return self.GD*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.GD*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def interpolation(self, u, integral=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        p = self.p
        ipoint = self.dof.interpolation_points()
        uI = self.function() 
        uI[:NN+(p-1)*NE, :] = u(ipoint)
        if p > 1:
            phi = self.vsmspace.basis

            def f(x, cellidx):
                return np.einsum('ijk, ij...k->ij...', u(x), phi(x, cellidx=cellidx, p=p-2))
            
            bb = integral(f, celltype=True)/self.vsmspace.area[..., np.newaxis, np.newaxis]
            uI[NN+(p-1)*NE:, :] = bb.reshape(-1, 2)
        return uI

    def function(self):
        return Function(self, self.GD)

    def array(self, dim):
        gdof = self.dof.number_of_global_dofs() # 获得对应标量空间的全局自由度
        shape = (gdof, dim)
        return np.zeros(shape, dtype=np.float)


        


