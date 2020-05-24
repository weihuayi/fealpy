import numpy as np
from numpy.linalg import inv
from .function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolyhedronMeshIntegralAlg
from ..quadrature import FEMeshIntegralAlg
from ..common import ranges


class SMDof3d():
    """
    缩放单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p # 默认的空间次数
        self.multiIndex = self.multi_index_matrix() # 默认的多重指标
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def multi_index_matrix(self, p=None):
        """
        Compute the natural correspondence from the one-dimensional index
        starting from 0.

        Notes
        -----

        0<-->(0, 0), 1<-->(1, 0), 2<-->(0, 1), 3<-->(2, 0), 4<-->(1, 1),
        5<-->(0, 2), .....

        """
        p = self.p if p is None else p
        ldof = (p+1)*(p+2)*(p+3)//6
        idx = np.arange(1, ldof)
        idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
        idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
        idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
        idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
        multiIndex = np.zeros((ldof, 4), dtype=np.int)
        multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
        multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
        multiIndex[1:, 1] = idx0 - idx2
        multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
        return multiIndex

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p)
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self, p=None):
        p = self.p if p is None else p
        return (p+1)*(p+2)*(p+3)//6

    def number_of_global_dofs(self, p=None):
        ldof = self.number_of_local_dofs(p=p)
        NC = self.mesh.number_of_cells()
        return NC*ldof

class ScaledMonomialSpace3d():
    def __init__(self, mesh, p, q=None, bc=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.p = p
        self.mesh = mesh
        self.cellbarycenter = mesh.entity_barycenter('cell') if bc is None else bc
        self.cellmeasure = mesh.entity_measure('cell')
        self.cellsize = self.cellmeasure**(1/3)
        self.dof = SMDof3d(mesh, p)
        self.GD = 3

        q = q if q is not None else p+3

        mtype = mesh.meshtype
        if mtype in {'polyhedron'}:
            self.integralalg = PolyhedronMeshIntegralAlg(
                    self.mesh, q,
                    cellmeasure=self.cellmeasure,
                    cellbarycenter=self.cellbarycenter)
        elif mtype in  {'tet'}:
            self.integralalg = FEMeshIntegralAlg(
                    self.mesh, q,
                    cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
    
    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self, p=None):
        return self.dof.cell_to_dof(p=p)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def number_of_local_dofs(self, p=None):
        return self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)

    def basis(self, point, index=None, p=None):
        """
        Compute the basis values at point

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., NC, 2), NC is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., NC, ldof)

        """
        p = self.p if p is None else p
        h = self.cellsize
        NC = self.mesh.number_of_cells()

        ldof = self.number_of_local_dofs(p=p)
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=self.ftype)  # (..., M, ldof)
        index = index if index is not None else np.s_[:] 
        phi[..., 1:3] = (point - self.cellbarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 4
            for i in range(2, p+1):
                n = (i+1)*i//2
                phi[..., start:start+n] = phi[..., start-n:start]*phi[..., [1]]
                phi[..., start+n:start+n+i] = phi[..., start-i:start]*phi[..., [2]]
                phi[..., start+n+i] = phi[..., start-1]*phi[..., 3]
                start += n + i + 1  
        return phi

    def grad_basis(self, point, index=None, p=None):

        p = self.p if p is None else p
        h = self.cellsize
        num = len(h) if index is  None else len(index)
        index = np.s_[:] if index is None else index 

        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 2)
        phi = self.basis(point, index=index, p=p-1)
        idx = self.index1(p=p)
        gphi = np.zeros(shape, dtype=np.float)
        xidx = idx['x']
        yidx = idx['y']
        gphi[..., xidx[0], 0] = np.einsum('i, ...i->...i', xidx[1], phi) 
        gphi[..., yidx[0], 1] = np.einsum('i, ...i->...i', yidx[1], phi)
        if point.shape[-2] == num:
            return gphi/h[index].reshape(-1, 1, 1)
        elif point.shape[0] == num:
            return gphi/h[index].reshape(-1, 1, 1, 1)
