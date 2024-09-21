from builtins import callable, str

class BoundaryConditions:
    def __init__(self, 
                 force_func: callable, 
                 dirichlet_func: callable, 
                 is_dirichlet_boundary_edge_func: callable, 
                 is_dirichlet_node_func: callable,  # 新增的参数
                 is_dirichlet_direction_func: callable):
        """
        Initialize boundary conditions properties.

        Args:
            force_func (callable): A function defining the external load applied to the structure.
            dirichlet_func (callable): A function defining the Dirichlet boundary conditions.
            is_dirichlet_boundary_edge_func (callable): A function that determines which boundary edges 
                satisfy the Dirichlet condition.
            is_dirichlet_node_func (callable): A function that determines which nodes on the boundary edge 
                satisfy the Dirichlet condition.
            is_dirichlet_direction_func (callable): A function that determines which component of the degrees of 
                freedom is fixed.
        """
        self.force_func = force_func
        self.dirichlet_func = dirichlet_func
        self.is_dirichlet_boundary_edge_func = is_dirichlet_boundary_edge_func
        self.is_dirichlet_node_func = is_dirichlet_node_func
        self.is_dirichlet_direction_func = is_dirichlet_direction_func

    def force(self, points):
        """
        Compute the load value at the given points.

        Args:
            points: The coordinates of the points where the load is applied.

        Returns:
            The load values at the given points.
        """
        return self.force_func(points)

    def dirichlet(self, points):
        """
        Compute the Dirichlet boundary condition values at the given points.

        Args:
            points: The coordinates of the points where the Dirichlet condition is applied.

        Returns:
            The Dirichlet boundary condition values at the given points.
        """
        return self.dirichlet_func(points)

    def is_dirichlet_boundary_edge(self, points):
        """
        Determine if the given boundary edges satisfy the Dirichlet condition.

        Args:
            points: The coordinates of the points defining the boundary edges.

        Returns:
            Boolean values indicating if the boundary edges satisfy the Dirichlet condition.
        """
        return self.is_dirichlet_boundary_edge_func(points)
    
    def is_dirichlet_node(self):
        """
        Determine if the nodes on the boundary edge satisfy the Dirichlet condition.

        Returns:
            Boolean values indicating if the nodes satisfy the Dirichlet condition.
        """
        return self.is_dirichlet_node_func()

    def is_dirichlet_direction(self):
        """
        Determine which component of the degrees of freedom is fixed.

        Returns:
            The component of the degrees of freedom that is fixed by the Dirichlet condition.
        """
        return self.is_dirichlet_direction_func()

    def __repr__(self) -> str:
        """
        Return a string representation of the boundary conditions.

        Returns:
            str: A string showing the functions defining the boundary conditions.
        """
        return ("BoundaryConditions(force_func, dirichlet_func, "
                "is_dirichlet_boundary_edge_func, is_dirichlet_node_func, "
                "is_dirichlet_direction_func)")
