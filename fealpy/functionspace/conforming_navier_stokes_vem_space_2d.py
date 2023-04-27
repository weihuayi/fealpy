import numpy as np
from numpy.linalg import inv

from .Function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d


class ConformingNavierStokesVEMSpace2d:

    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        self.dof = CNSVEMDof2d(mesh, p)


class CNSVEMDof2d:
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof] = True
        return isBdDof

    def edge_to_dof(self):
        pass

    def cell_to_dof(self):
        pass

    def number_of_global_dofs(self):
        pass

    def number_of_local_dofs(self):
        pass
