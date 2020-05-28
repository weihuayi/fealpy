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
    三维缩放单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p # 默认的空间次数
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cdof = self.number_of_local_dofs(p=p, etype='cell')
        cell2dof = np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof

    def number_of_local_dofs(self, p=None, etype='cell'):
        p = self.p if p is None else p
        if etype in {'cell', 3}:
            return (p+1)*(p+2)*(p+3)//6
        elif etype in {'face', 2}:
            return (p+1)*(p+2)//2
        elif etype in {'edge', 1}:
            return p+1

    def number_of_global_dofs(self, p=None, etype='cell'):
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p, etype=etype)
        return NC*ldof


class ScaledMonomialSpace3d():
    def __init__(self, mesh, p, q=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.p = p
        self.GD = 3
        self.mesh = mesh
        self.dof = SMDof3d(mesh, p)

        self.cellbarycenter = mesh.entity_barycenter('cell')
        self.facebarycenter = mesh.entity_barycenter('face')

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

        self.cellmeasure = self.integralalg.cellmeasure 
        self.facemeasure = self.integralalg.facemeasure 
        self.edgemeasure = self.integralalg.edgemeasure

        self.cellsize = self.cellmeasure**(1/3)
        self.facesize = self.facemeasure**(1/2)
        self.edgesize = self.edgemeasure

        # get the axis frame on the face by svd
        n = mesh.face_unit_normal()
        _, _, frame = np.linalg.svd(n[:, np.newaxis, :]) 
        frame[:, 0, :] = n

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
        Compute the basis values at point in cell

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., NC, 2), NC is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., NC, cdof)

        """
        p = self.p if p is None else p
        h = self.cellsize
        cdof = self.number_of_local_dofs(p=p, etype='cell')
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

    def face_basis(self, point, index=None, p=None):
        """
        Compute the basis values at point on each face 

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., NF, 3), NC is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., NF, fdof)

        """
        p = self.p if p is None else p
        h = self.facesize

        fdof = self.number_of_local_dofs(p=p, etype='face')
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float)  # (..., NF, ldof)
        index = index if index is not None else np.s_[:] 
        phi[..., 1:3] = (point - self.facebarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    def grad_basis(self, point, index=None, p=None):
        pass

    def cell_mass_matrix(self):
        pass

    def face_mass_matrix(self):
        pass
