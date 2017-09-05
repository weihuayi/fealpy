
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

def source(f, V, qf):
    mesh = V.mesh
    NC = mesh.number_of_cells()
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    cell2dof = V.cell_to_dof()
    
    bb = np.zeros((NC, ldof), dtype=self.dtype)
    area = mesh.area()
    for i in range(nQuad):
        lambda_k, w_k = qf.get_gauss_point_and_weight(i)
        p = mesh.bc_to_point(lambda_k)
        fval = f(p)
        phi = V.basis(lambda_k)
        for j in range(ldof):
            bb[:, j] += fval*phi[j]*w_k
    bb *= area.reshape(-1, 1)
    cell2dof = V.cell_to_dof()
    b = np.zeros((gdof,), dtype=self.dtype)
    np.add.at(b, cell2dof.flatten(), bb.flatten())
    #b = np.bincount(cell2dof.flatten(), weights=bb.flatten(), minlength=gdof)
    return b 

def laplace(V, qf, a=None):
    mesh = V.mesh
    NC = mesh.number_of_cells() 
    area = mesh.area()
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    cell2dof = V.cell_to_dof()

    A = coo_matrix((gdof, gdof), dtype=self.dtype)
    for i in range(ldof):
        for j in range(i,ldof):
            val = np.zeros((NC,), dtype=self.dtype)
            for q in range(nQuad):
                lambda_k, w_k = qf.get_gauss_point_and_weight(q)
                gradphi = V.grad_basis(lambda_k)
                val += np.sum(gradphi[:,i,:]*gradphi[:,j,:], axis=1)*w_k
            A += coo_matrix((val*area, (cell2dof[:,i], cell2dof[:,j])), shape=(gdof, gdof))
            if j != i:
                A += coo_matrix((val*area, (cell2dof[:,j], cell2dof[:,i])), shape=(gdof, gdof))
    return A.tocsr()
