import numpy as np


class BilinearForm:
    def __init__(self):
        pass

    def get_matrix(self):
        pass

class EllipticForm(BilinearForm):
    def __init__(self, V, 
            dc=1.0, 
            ac=None, 
            rc=None,
            qf=1, dtype=np.float):
        self.V = V
        self.qf = qf

        self.dc = dc
        self.ac = ac
        self.rc = rc

        self.dtype = dtype

    def get_matrix(self):
        if 


class DiffusionSymetricForm(BilinearForm):
    def __init__(self, V, diffusionCoefficient=None,
            qf=1, dtype=np.float):
        self.V = V
        self.qf = qf 
        self.dtype = dtype

    def get_matrix(self):
        V = self.V

        mesh = V.mesh
        NC = mesh.number_of_cells() 
        area = mesh.area()
        
        quadFormula = TriangleQuadrature(self.qf)
        nQuad = quadFormula.get_number_of_quad_points()

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        A = coo_matrix((gdof, gdof), dtype=self.dtype)

        for i in range(ldof):
            for j in range(i,ldof):
                val = np.zeros((NC,), dtype=self.dtype)
                for q in range(nQuad):
                    lambda_k, w_k = quadFormula.get_gauss_point_and_weight(q)
                    gradphi = V.grad_basis(lambda_k)
                    val += np.sum(gradphi[:,i,:]*gradphi[:,j,:], axis=1)*w_k
                A += coo_matrix((val*area, (cell2dof[:,i], cell2dof[:,j])), shape=(gdof, gdof))
                if j != i:
                    A += coo_matrix((val*area, (cell2dof[:,j], cell2dof[:,i])), shape=(gdof, gdof))

        return A.tocsr()
