
from typing import Union, Optional, Callable
import numpy as np

from .Function import Function
from ..decorator import barycentric 

from ..mesh import TriangleMesh, TetrahedronMesh

class CrouzeixRaviartFESpace():
    def __init__(self, 
            mesh: Union[TriangleMesh, TetrahedronMesh],
            ):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')

        self.dof = CRDof(mesh)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.itype = mesh.itype
        self.ftype = mesh.ftype


    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def is_boundary_dof(self, threshold=None):
        return self.dof.is_boundary_dof(threshold=threshold) 

    @barycentric
    def basis(self, bc):
        """
        @brief 
        """
        phi = 1 - self.GD*bc
        return phi[..., None, :] # (..., 1, ldof)

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """
        @brief 
        """
        mesh = self.mesh
        gphi = -self.GD*mesh.grad_lambda(inde=index) # (NC, TD+1, GD)
        if len(bc.shape) == 1:
            return gphi 
        else:
            return gphi[None, ...] # (NQ, NC, TD+1, GD) 多个积分点, 增加一个轴

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        index = index if index is not None else np.s_[:]
        val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

class CRDof():
    def __init__(self, mesh):
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof()

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        mesh = self.mesh
        NF = mesh.number_of_faces()
        return np.arange(NF).reshape(-1, 1)

    def cell_to_dof(self, index=np.s_[:]):
        mesh = self.mesh
        cell2dof = mesh.ds.cell_to_face()
        return cell2dof[index]

    def interpolation_points(self):
        mesh = self.mesh
        ipoint = mesh.entity_barycenter('face')
        return ipoint

    def number_of_global_dofs(self):
        gdof = self.mesh.number_of_faces()
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        mesh = self.mesh        
        TD = mesh.top_dimension()
        if doftype in {'cell'}:
            return TD+1 
        elif doftype in {'face'}:
            return 1
        elif doftype in {'node'}:
            return 0
