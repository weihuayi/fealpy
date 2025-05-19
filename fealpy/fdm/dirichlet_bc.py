from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import spdiags

class DirichletBC:
    """Class for applying Dirichlet boundary conditions

    This class modifies a linear system (e.g., from finite difference or finite volume methods)
    to incorporate Dirichlet boundary conditions. It supports applying boundary values on
    either the entire boundary or only a subset defined by a user-provided threshold function.

    Parameters
    ----------
    mesh : Mesh
        The mesh object containing node coordinates and boundary information.
    gd : Callable
        The Dirichlet boundary function. Accepts node coordinates and returns values.
    threshold : Callable, optional, default=None
        A function that selects a subset of boundary nodes. It takes an array of boundary
        node coordinates (shape: (N, 2)) and returns a boolean array indicating which
        nodes should have Dirichlet conditions applied.

    Attributes
    ----------
    mesh : Mesh
        The mesh associated with the problem domain.
    gd : Callable
        The Dirichlet boundary value function.
    threshold : Callable or None
        A condition function for selecting specific boundary nodes.

    Methods
    -------
    apply(A, f, uh=None)
        Apply the Dirichlet boundary conditions to the linear system and return the modified system.
    """

    def __init__(self, mesh, gd, threshold=None):
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def apply(self, A, f, uh=None):
        """Apply Dirichlet boundary conditions to the linear system

        This function modifies the matrix `A` and right-hand side vector `f`
        so that the solution `uh` satisfies the given Dirichlet boundary conditions.

        Parameters
        ----------
        A : SparseMatrix
            The system matrix of shape (N, N), where N is the number of nodes.
        f : TensorLike
            The right-hand side vector of shape (N,).
        uh : TensorLike, optional, default=None
            The current approximation of the solution. If None, initialized to zero.

        Returns
        -------
        A_new : SparseMatrix
            The modified system matrix, where Dirichlet rows are replaced with identity.
        f_new : TensorLike
            The modified right-hand side vector with boundary values applied.

        Notes
        -----
        - If `threshold` is not provided, boundary conditions are applied on all boundary nodes.
        - If `threshold` is provided, it will be used to select a subset of boundary nodes.
        - The matrix is modified such that at boundary nodes:
        
          .. math:: u_i = g(x_i)

          while the system remains unchanged for interior nodes using projection matrices.

        Examples
        --------
        >>> def bd_diag(node):
        >>>     return abs(node[:, 0] - node[:, 1]) < 1e-10
        >>>
        >>> bc = DirichletBC(mesh, gd=exact_solution, threshold=bd_diag)
        >>> A_bc, f_bc = bc.apply(A, f)
        """
        if uh is None:
        # Initialize uh as a zero vector if not provided
            if hasattr(A, 'values_context'):
                uh = bm.zeros(A.shape[0], **A.values_context())
            else:
                uh = bm.zeros(A.shape[0], dtype=A.dtype)
        # Get all node coordinates
        node = self.mesh.entity('node')

        if self.threshold is None:
            # If no threshold function is provided, apply BCs to all boundary nodes
            bdFlag = self.mesh.boundary_node_flag()
        else:
            # Apply BCs only to selected boundary nodes based on threshold
            total_bd_idx = self.mesh.boundary_node_index()   # Indices of all boundary nodes
            bd_node = self.mesh.node[total_bd_idx]           # Coordinates of boundary nodes
            mark = self.threshold(bd_node)                   # Boolean array indicating selected nodes
            bdFlag = bm.zeros(self.mesh.number_of_nodes(), dtype=bool)
            bdFlag[total_bd_idx[mark]] = True                # Mark selected boundary nodes as True

        # Assign boundary values at selected nodes
        uh = bm.set_at(uh, bdFlag, self.gd(node[bdFlag]))

        # Update right-hand side to account for boundary values
        f = f - A @ uh
        f = bm.set_at(f, bdFlag, uh[bdFlag])

        # Construct projection matrices
        D0 = spdiags(1 - bdFlag, 0, A.shape[0], A.shape[0])  # Keeps interior equations
        D1 = spdiags(bdFlag, 0, A.shape[0], A.shape[0])      # Identity on boundary nodes

        # Apply boundary conditions to the matrix
        A = D0 @ A @ D0 + D1
        return A, f

