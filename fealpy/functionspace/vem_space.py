"""
Virtual Element Space

"""
import numpy as np
from ..common import ranges
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
        p = self.p
        ldof = self.number_of_local_dofs() 
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex 

    def cell_to_dof(self):
        p = self.p
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
        self.area = mesh.cell_area()
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
        point : numpy array
            The shape of point is (..., M, 2) 
        """
        if p is None:
            p = self.p
        h = self.h
        NC = self.mesh.number_of_cells()

        ldof = self.number_of_local_dofs(p=p) 
        if p == 0: 
            return np.ones(point.shape[:-1], dtype=np.float) 
    
        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float) # (..., M, ldof)
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
                hphi[..., start+2:start+i+1, 1] = np.eisum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                r0 = np.arange(1, i)
                r0 = r0*r0[-1::-1]
                hphi[..., start+1:start+i, 2] = np.eisum('i, ...i->...i', r0, phi[..., start-2*i+1:start-i])
                start += i+1

        if cellidx is None:
            return hphi/area.reshape(-1, 1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            return hphi/area[cellidx].reshape(-1, 1, 1)
        
    def laplace_value(self, uh, point, cellidx=None):
        lphi = self.laplace_basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellIdx is None:
            return np.einsum('ij, ...ij->...i', uh[cell2dof], lphi)
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ij->...i', uh[cell2dof[cellIdx]], lphi)

    def function(self):
        f = FiniteElementFunction(self)
        return f 

    def array(self, dim=None):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        return np.zeros(NC*ldof, dtype=np.float)

    def number_of_local_dofs(self, p=None):
        return self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

class VEMDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

    def boundary_dof(self):
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        edge2dof = self.edge_to_dof()
        isBdEdge = self.mesh.ds.boundary_edge_flag()
        isBdDof[edge2dof[isBdEdge]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE= mesh.number_of_edges()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int) 
        edge2dof[:, [0, p]] = edge 
        if p > 1:
            edge2dof[:, 1:-1] = np.arange(NN, NN + NE*(p-1)).reshape(NE, p-1)
        return edge2dof
    
    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        cellLocation = mesh.ds.cellLocation

        if p == 1:
            return cell, cellLocation
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = np.zeros(NC+1, dtype=np.int)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)
            
            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()

            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
            cell2dof[idx] = edge2dof[:, 0:p]
            
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
            cell2dof[idx] = edge2dof[isInEdge, p:0:-1]

            NN = mesh.number_of_nodes()
            NV = mesh.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + np.arange(idof)
            cell2dof[idx] = N + NE*(p-1) + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof, cell2dofLocation


    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        gdof = NN
        p = self.p
        if p > 1:
            gdof += NE*(p-1) + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        ldofs = NV
        p = self.p
        if p > 1:
            ldofs += NV*(p-1) + (p-1)*p//2
        return ldofs

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == 1:
            return node 
        if p > 1:
            NN = mesh.number_of_nodes()
            GD = mesh.geo_dimension() 
            NE = mesh.number_of_edges()

            gdof = self.number_of_global_dofs()
            ipoint = np.zeros((gdof, GD), dtype=np.float)
            ipoint[:NN, :] = node 
            edge = mesh.entity('edge')

            qf = GaussLobattoQuadrature(p + 1)

            bcs = qf.quadpts[1:-1, :]
            ipoint[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
            return ipoint


class VirtualElementSpace2d():
    def __init__(self, mesh, p = 1):
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.dof = VEMDof2d(mesh, p)

    def cell_to_dof(self):
        return self.dof.cell2dof, self.dof.cell2dofLocation

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def basis(self, bc):
        pass

    def grad_basis(self, bc):
        pass

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass
    
    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

    def function(self, dim=None):
        return FiniteElementFunction(self, dim=dim)

    def interpolation(self, u, integral=None):

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        p = self.p
        ipoint = self.interpolation_points()
        uI = self.function() 
        uI[:NN+(p-1)*NE] = u(ipoint[:NN+(p-1)*NE])
        if p > 1:
            phi = self.smspace.basis

            def f(x, cellidx):
                return np.einsum('ij, ij...->ij...', u(x), phi(x, cellidx=cellidx, p=p-2))
            
            if p == 2:
                bb = integral(f, celltype=True)/self.smspace.area
            else:
                bb = integral(f, celltype=True)/self.smspace.area[..., np.newaxis]
            uI[NN+(p-1)*NE:] = bb.reshape(-1)
        return uI

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def projection(self, u, up):
        pass

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

