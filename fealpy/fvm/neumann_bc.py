from fealpy.sparse import spdiags
from fealpy.backend import backend_manager as bm


class NeumannBC:
    
    def __init__(self, mesh, gd, threshold=None):
        
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def ThresholdApply(self, A, f, uh=None):
        """
        Apply Dirichlet boundary conditions to selected boundary cells based on a threshold.

        This method modifies the system matrix `A` and right-hand side vector `f` by applying 
        Dirichlet boundary conditions to cells selected by the threshold function. It supports 
        selective boundary condition application based on coordinate criteria.

        Args:
            A (sparse matrix): System matrix to be modified.
            f (ndarray): Right-hand side vector to be modified.
            uh (ndarray, optional): Solution vector to store boundary values. If None, initialized as zeros.

        Returns:
            tuple: (A, f)
                - A (sparse matrix): Modified system matrix with boundary conditions applied.
                - f (ndarray): Modified right-hand side vector with boundary contributions.

        Raises:
            ValueError: If threshold is not a callable function.
        """
        total_bd_idx = self.mesh.boundary_cell_index()
        points = self.mesh.entity_barycenter('cell')
        NC = self.mesh.number_of_cells()
        bd_node = points[total_bd_idx]
        if callable(self.threshold):
            try:
                # Try applying condition to x-coordinate only
                x = bd_node[:, 0]
                bd_idx = self.threshold(x)
                bd_idx = bm.array(bd_idx, dtype=bm.bool)
                if not bm.any(bd_idx):  # Check if bd_idx is all False
                    y = bd_node[:, 1]
                    bd_idx = self.threshold(y)
                    bd_idx = bm.array(bd_idx, dtype=bm.bool)
            except Exception:
                # Fall back to applying condition to full node coordinates
                bd_idx = self.threshold(bd_node)
                bd_idx = bm.array(bd_idx, dtype=bm.bool)
        else:
            raise ValueError("self.threshold must be a callable (e.g., lambda x: (x==0.5)|(x==2.5) or a function).")
        index = total_bd_idx[bd_idx]
        bdFlag_u = bm.zeros(NC)
        bdFlag_u[index] = 1
        D0 = spdiags(1 - bdFlag_u, 0, A.shape[0], A.shape[0])  # Keeps interior equations
        D1 = spdiags(bdFlag_u, 0, A.shape[0], A.shape[0])      # Identity on boundary nodes
        # Apply boundary conditions to the matrix
        if uh is None:
            # Initialize uh as a zero vector if not provided
            if hasattr(A, 'values_context'):
                uh = bm.zeros(A.shape[0], **A.values_context())
            else:
                uh = bm.zeros(A.shape[0], dtype=A.dtype)
        uh = bm.set_at(uh, index, self.gd(points[index]))
        f = f - A @ uh
        f = bm.set_at(f, index, uh[index])
        A = D0.matmul(A.matmul(D0)) + D1
        return A, f
    
    def DiffusionApply(self,f):
        
        bdedge = self.mesh.boundary_face_index()
        points = self.mesh.entity_barycenter('face')[bdedge, :]
        neumann = self.gd(points)
        e2c = self.mesh.edge_to_cell()
        bm.add_at(f, e2c[bdedge,1], neumann*self.mesh.entity_measure('face')[bdedge])        
        return f

    def ConvectionApplyX(self,A,b):

        NC = self.mesh.number_of_cells()
        bdIdx = bm.zeros(NC)
        Sf = self.mesh.edge_normal()
        bdedge = self.mesh.boundary_face_index()
        e2c = self.mesh.edge_to_cell()
        bde2c = e2c[bdedge, 0]
        bm.add_at(bdIdx, bde2c, Sf[bdedge, 0])
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0

        cell_measure = self.mesh.entity_measure('cell')
        face_measure = self.mesh.entity_measure('face')
        LNE = self.mesh.number_of_vertices_of_cells()
        d = 2*cell_measure[e2c[bdedge, 0]]/(LNE*face_measure[bdedge])
        gf = self.gd(self.mesh.entity_barycenter('face')[bdedge, :])
        bm.add_at(b, bde2c, -gf * d * Sf[bdedge, 0])
        
        return A,b

    def ConvectionApplyY(self,A,b):

        NC = self.mesh.number_of_cells()
        bdIdx = bm.zeros(NC)
        Sf = self.mesh.edge_normal()
        bdedge = self.mesh.boundary_face_index()
        e2c = self.mesh.edge_to_cell()
        bde2c = e2c[bdedge, 0]
        bm.add_at(bdIdx, bde2c, Sf[bdedge, 1])
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0

        cell_measure = self.mesh.entity_measure('cell')
        face_measure = self.mesh.entity_measure('face')
        LNE = self.mesh.number_of_vertices_of_cells()
        d = 2*cell_measure[e2c[bdedge, 0]]/(LNE*face_measure[bdedge])
        gf = self.gd(self.mesh.entity_barycenter('face')[bdedge, :])
        bm.add_at(b, bde2c, -gf * d * Sf[bdedge, 1])
        
        return A,b

    