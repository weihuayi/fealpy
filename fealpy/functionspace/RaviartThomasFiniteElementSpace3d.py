import numpy as np

from .function import Function

from .femdof import multi_index_matrix1d
from .femdof import multi_index_matrix2d
from .femdof import multi_index_matrix3d

class RaviartThomasFiniteElementSpace3d:
    def __init__(self, mesh, p=0, q=None):
        self.p = p
        self.mesh = mesh

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.multi_index_matrix = [multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]

    def basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()

        ldof = self.number_of_local_dofs()

        p = self.p
        phi = np.zeors((NC, ldof, dim), dtype=self.dtype)


        return phi

    def grad_basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        p = self.p

        gradPhi = np.zeros((NC, ldof, dim, dim), dtype=self.dtype)

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        divPhi = np.zeors((NC, ldof), dtype=self.dtype)

        return divPhi 

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 0:
            return NE
        elif p==1:
            return 2*NE
        else:
            #TODO: raise a error
            print("error!")


    def number_of_local_dofs(self):
        p = self.p
        if p==0:
            return 3
        elif p==1:
            return 6
        else:
            #TODO: raise a error
            print("error!")
