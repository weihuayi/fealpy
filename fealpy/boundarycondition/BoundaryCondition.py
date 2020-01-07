import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

class BoundaryCondition():

    def __init__(self, space, dirichlet=None, neuman=None, robin=None):
        self.space = space
        self.dirichlet = dirichlet
        self.neuman = neuman
        self.robin = robin

    def apply_dirichlet_bc(self, A, b, uh=None, dim=None, is_dirichlet_boundary=None):
        if uh is None:
            uh = self.space.function(dim=dim)
            if self.dirichlet is not None:
                self.space.set_dirichlet_bc(uh, self.dirichlet,
                        is_dirichlet_boundary=is_dirichlet_boundary)
                b -= A@uh
                bdIdx = np.zeros(gdof, dtype=np.int)
                bdIdx[isDDof] = 1
                Tbd = spdiags(bdIdx, 0, gdof, gdof)
                T = spdiags(1-bdIdx, 0, gdof, gdof)
                A = T@A@T + Tbd
                b[isDDof] = uh[isDDof]
                return A, b

    def apply_neuman_bc(self, b, dim=None, is_neuman_boundary=None):
        space = self.space
        TD = space.top_dimension()
        neuman = self.neuman
        mesh = space.mesh
        face = mesh.entity('face')
        bc = mesh.entity_barycenter('face')
        node = mesh.entity('node')
        isNFace = is_neuman_boundary(bc)
        bdface = face[isNFace]

        # the unit outward normal on boundary edge
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (node[bdEdge[:,1],] - node[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape((-1,1))

class DirichletBC:
    def __init__(self, V, g0, is_dirichlet_dof=None):
        self.V = V
        self.g0 = g0

        if is_dirichlet_dof == None:
            isBdDof = V.boundary_dof()
        else:
            ipoints = V.interpolation_points()
            isBdDof = is_dirichlet_dof(ipoints)

        self.isBdDof = isBdDof

    def apply(self, A, b):
        """ Modify matrix A and b
        """
        g0 = self.g0
        V = self.V
        isBdDof = self.isBdDof

        gdof = V.number_of_global_dofs()
        x = np.zeros((gdof,), dtype=np.float)
        ipoints = V.interpolation_points()
        # the length of ipoints and isBdDof maybe different
        idx, = np.nonzero(isBdDof)
        x[isBdDof] = g0(ipoints[idx])
        b -= A@x
        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        A = T@A@T + Tbd

        b[isBdDof] = x[isBdDof]
        return A, b

    def apply_on_matrix(self, A):

        V = self.V
        isBdDof = self.isBdDof
        gdof = V.number_of_global_dofs()

        bdIdx = np.zeros((A.shape[0], ), np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd

        return A

    def apply_on_vector(self, b, A):
        
        g0 = self.g0
        V = self.V
        isBdDof = self.isBdDof

        gdof = V.number_of_global_dofs()
        x = np.zeros((gdof,), dtype=np.float)

        ipoints = V.interpolation_points()
        x[isBdDof] = g0(ipoints[isBdDof,:])
        b -= A@x

        b[isBdDof] = x[isBdDof] 

        return b



        


