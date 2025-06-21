import numpy as np
from .LagrangeFiniteElementSpace2d import LagrangeFiniteElementSpace2d

class LagrangeFiniteElementVectorSpace2d:

    def __init__(self, mesh, p=1, dtype=np.float):
        self.scalarspace = LagrangeFiniteElementSpace2d(mesh, p, dtype=dtype)
        self.mesh = mesh
        self.p = p 
        self.dtype=dtype

    def basis(self, bc):
        return self.scalarspace.basis(bc)

    def grad_basis(self, bc):
        return self.scalarspace.grad_basis(bc)

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        mesh = self.scalarspace.mesh
        NC = mesh.number_of_cells()
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        val = np.zeros((NC,2), dtype=self.dtype)
        val[:,0] = (uh[cell2dof,0]@phi.reshape((-1,1))).reshape((-1,))
        val[:,1] = (uh[cell2dof,1]@phi.reshape((-1,1))).reshape((-1,))
        return val 

    def grad_value(self, uh, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        val = np.zeros((NC,2,2), dtype=self.dtype)
        val[:,0,0] = (uh[cell2dof,0]*gradphi[:,:,0]).sum(axis=1)
        val[:,0,1] = (uh[cell2dof,0]*gradphi[:,:,1]).sum(axis=1)
        val[:,1,0] = (uh[cell2dof,1]*gradphi[:,:,0]).sum(axis=1)
        val[:,1,1] = (uh[cell2dof,1]*gradphi[:,:,1]).sum(axis=1)
        return val

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        val = np.zeros((NC,), dtype=self.dtype)
        val += (uh[cell2dof,0]*gradphi[:,:,0]).sum(axis=1)
        val += (uh[cell2dof,1]*gradphi[:,:,1]).sum(axis=1)
        return val

    def number_of_global_dofs(self):
        return self.scalarspace.number_of_global_dofs()
        
    def number_of_local_dofs(self):
        return self.scalarspace.number_of_local_dofs()

    def cell_to_dof(self):
        return self.scalarspace.cell_to_dof()

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,2),dtype=self.dtype)

