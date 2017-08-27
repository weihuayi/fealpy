
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

class LaplaceSymetricForm:
    def __init__(self, V, dtype=np.float):
        self.V = V
        self.dtype = dtype

    def get_matrix(self):
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()

        p = V.p
        area = V.smspace.area
        h = V.smspace.h

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        smldof = V.smspace.number_of_local_dofs()

        A = coo_matrix((gdof, gdof), dtype=self.dtype)

        cell2dof, cell2dofLocation = V.cell_to_dof()
        cell = mesh.ds.cell
        point = mesh.point

        if p == 1:
            NV = mesh.number_of_vertices_of_cells()
            B = np.zeros((smldof, cell2dof.shape[0]), dtype=self.dtype) 
            B[0, :] = 1/np.repeat(NV, NV)
            B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1,-1)

            bc = np.repeat(V.smspace.barycenter, NV, axis=0)
            D = np.ones((cell2dof.shape[0], smldof), dtype=self.dtype)
            D[:, 1:] = (point[cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            G = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            for i in range(NC):
                K = B[:, cell2dofLocation[i]:cell2dofLocation[i+1]].T@G@B[:,
                        cell2dofLocation[i]:cell2dofLocation[i+1]]
                M = np.eye(ldof[i]) - D[cell2dofLocation[i]:cell2dofLocation[i+1], :]@B[:, cell2dofLocation[i]:cell2dofLocation[i+1]]
                K += M.T @ M
                dof = cell[cell2dofLocation[i]:cell2dofLocation[i+1]]
                I = np.repeat(dof, ldof[i])
                J = np.repeat(dof.reshape(1, -1), ldof[i], axis=0).flatten()
                A += coo_matrix((K.flatten(), (I, J)), shape=(gdof, gdof), dtype=self.dtype)
        else:
            cell2dof, cell2dofLocation = V.cell_to_dof()

        return A.tocsr()
            

class SourceForm:
    def __init__(self, V, f, dtype=np.float):
        self.V = V
        self.f = f
        self.dtype = dtype

    def get_vector(self):
        V = self.V
        mesh = V.mesh

        ldof = V.number_of_local_dofs()
        bb = np.zeros(ldof.sum(), dtype=self.dtype)
        point = mesh.point
        NV = mesh.number_of_vertices_of_cells()
        F = self.f(point)
        area = V.smspace.area
        cell2dof , cell2dofLocation = V.cell_to_dof()
        bb = F[cell2dof]/np.repeat(NV, NV)*np.repeat(area, NV)

        gdof = V.number_of_global_dofs()
        b = np.bincount(cell2dof, weights=bb, minlength=gdof)
        return b
