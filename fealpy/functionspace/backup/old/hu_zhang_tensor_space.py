
from .function import FiniteElementFunction
from .lagrange_fem_space import LagrangeFiniteElementSpace

class HuZhangTensorSpace2d():
    def __init__(self, mesh, p=3):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.T = np.array([[(1, 0), (0, 0)], [(0, 1), (1, 0)], [(0, 0), (0, 1)]])
    
    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return 3*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return 3*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bc, cellidx=None):
        mesh = self.mesh
        n, t = mesh.edge_frame()
        NE = mesh.number_of_edges()
        
        idx = np.array([(0, 0), (0, 1), (1, 1)])
        TT = np.zeros((NE, 3, 3), dtype=np.float)
        TT[:, 0, ...] = np.prod(t[:, idx], axis=-1) 
        TT[:, 1, ...] = np.sum(t[:, idx]*n[:, idx[:, [1, 0]]], axis=-1)/2
        TT[:, 2, ...] = np.prod(n[:, idx], axis=-1) 

        NC = mesh.number_of_cells()
        if len(phi0.shape) == 1:
            shape0 = (NC,)+ phi0.shape + (3, 2, 2)
            shape1 = (NC, -1, 2, 2)
        elif len(phi0.shape) == 2:
            shape0 = (phi0.shape[0], NC, phi0.shape[1], 3, 2, 2)
            shape1 = (phi0.shape[0], NC, -1, 2, 2)
        else:
            shape0 = phi0.shape[:-1] + (NC, phi0.shape[-1], 3, 2, 2)
            shape1 = phi0.shape[:-1] + (NC, -1, 2, 2)
        phi = np.zeros(shape0, dtype=np.float)

        dof = self.scalarspace.dof
        multiIndex = dof.multiIndex 

        isEdgeDof = (multiIndex == 0) & (multiIndex != p)
        cell2edge = mesh.ds.cell_to_edge()
        phi0 = self.scalarspace.basis(bc)

        for i in range(3):
            VAL = np.einsum('ijk, kmn->ijmn', TT[cell2edge[:, i]], self.T)
            phi[..., isEdgeDof[:, i], :, :, :] = np.einsum('...k, ijmn->...ikjmn', phi0[..., isEdgeDof[:, i]], VAL) 

        isOtherDof = isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2]
        phi[..., isOtherDof, :, :, :] = np.einsum('...k, jmn, i->...ikjmn', phi0[..., isOtherDof], self.T, np.ones(NC))
        
        return phi.reshape(shape1)

    def div_basis(self, bc, cellidx=None):
        mesh = self.mesh
        n, t = mesh.edge_frame()
        NE = mesh.number_of_edges()
        
        idx = np.array([(0, 0), (0, 1), (1, 1)])
        TT = np.zeros((NE, 3, 3), dtype=np.float)
        TT[:, 0, ...] = np.prod(t[:, idx], axis=-1) 
        TT[:, 1, ...] = np.sum(t[:, idx]*n[:, idx[:, [1, 0]]], axis=-1)/2
        TT[:, 2, ...] = np.prod(n[:, idx], axis=-1) 

        NC = mesh.number_of_cells()
        if len(phi0.shape) == 1:
            shape0 = (NC,)+ phi0.shape + (3, 2)
            shape1 = (NC, -1, 2) 
        elif len(phi0.shape) == 2:
            shape0 = (phi0.shape[0], NC) + (phi0.shape[1], 3, 2)
            shape1 = (phi0.shape[0], NC, -1, 2)
        else:
            shape0 = phi0.shape[:-1] + (NC, phi0.shape[-1], 3, 2)
            shape1 = phi0.shape[:-1] + (NC, -1, 2)

        dphi = np.zeros(shape0, dtype=np.float)

        dof = self.scalarspace.dof
        multiIndex = dof.multiIndex 

        isEdgeDof = (multiIndex == 0) & (multiIndex != p)
        cell2edge = mesh.ds.cell_to_edge()
        gphi = self.scalarspace.grad_basis(bc)

        for i in range(3):
            VAL = np.einsum('ijk, kmn->ijmn', TT[cell2edge[:, i]], T)
            dphi[..., isEdgeDof[:, i], :, :] = np.einsum('...ikm, ijmn->...ikjn', gphi[..., isEdgeDof[:, i], :], VAL)

        isOtherDof = isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2]
        dphi[..., isOtherDof, 0, 0] = gphi[..., isOtherDof, 0]
        dphi[..., isOtherDof, 2, 1] = gphi[..., isOtherDof, 1]
        dphi[..., isOtherDof, 1, 0] = ghpi[..., isOtherDof, 1]
        dphi[..., isOtherDof, 1, 1] = gphi[..., isOtherDof, 0]
        #dphi[..., isOtherDof, :, :] = np.einsum('...ijk, mkn->...ijmn', gphi[..., isOtherDof, :, :], T) 

        return dphi.reshape(shape1)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        s = '...ijmn, ij->...imn'
        uh0 = uh.reshape(-1, 3)
        if cellidx is None:
            uh0 = uh0[cell2dof].reshape(-1)
        else:
            uh0 = uh0[cell2dof[cellidx]].reshape(-1)
        val = np.einsum(s, phi, uh0) 
        return val 

    def div_value(self, uh, bc, cellidx=None):
        dphi = self.div_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        uh0 = uh.reshape(-1, 3)
        if cellidx is None:
            uh0 = uh0[cell2dof].reshape(-1)
        else:
            uh0 = uh0[cell2dof[cellidx]].reshape(-1)
        s ='...ijm, ij->...im'
        val = np.einsum(s, dphi, uh0)
        return val

    def function(self, dim=None):
        f = FiniteElementFunction(self, dim=dim)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(shape, dtype=np.float)

