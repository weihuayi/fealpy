from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class PressureVesselDesign:
    """
    A class representing the pressure vessel design optimization problem.

    This class implements the pressure vessel design problem, which is a constrained
    optimization problem in engineering design. The goal is to minimize the total cost
    of manufacturing a pressure vessel while satisfying design constraints.

    Attributes:
        dim (int): The dimensionality of the problem (4 design variables).
        optimal_val (float): The known optimal value for this problem.
    """

    def __init__(self):
        """
        Initializes the PressureVesselDesign problem.

        Sets up the problem dimensions and known optimal value for validation.
        """
        self.dim = 4
        self.optimal_val = 5.8853327736E+03  

    def get_optimal(self) -> float:
        """
        Returns the known optimal value for this optimization problem.

        Returns:
            float: The optimal objective function value.
        """
        return self.optimal_val

    def get_dim(self) -> int:
        """
        Returns the dimensionality of the optimization problem.

        Returns:
            int: The number of design variables (4).
        """
        return self.dim
    
    def get_bounds(self) -> tuple:
        """
        Returns the lower and upper bounds for each design variable.

        Returns:
            tuple: A tuple containing two arrays:
                - Lower bounds: [0, 0, 10, 10]
                - Upper bounds: [99, 99, 200, 200]
        """
        return bm.array([0, 0, 10, 10]), bm.array([99, 99, 200, 200])
    
    def penalty(self, value: TensorLike) -> TensorLike:
        """
        Calculates the penalty term for constraint violations.

        Applies a quadratic penalty function with a large penalty factor to
        handle constraint violations in the optimization.

        Parameters:
            value (TensorLike): The constraint violation values.

        Returns:
            TensorLike: The calculated penalty values.
        """
        penalty_factor = 10e+6
        penalty = 0 + ((0 < value) * (value < 1)) * value + (value >= 1) * (value ** 2)
        return penalty_factor * penalty
    
    def evaluate(self, x: TensorLike) -> TensorLike:
        """
        Evaluates the objective function for given design variables.

        Computes the total cost of the pressure vessel design including
        penalty terms for constraint violations.

        Parameters:
            x (TensorLike): Design variable matrix of shape (N, 4), where:
                - x[:, 0]: Thickness of the shell (x1)
                - x[:, 1]: Thickness of the head (x2)
                - x[:, 2]: Inner radius (x3)
                - x[:, 3]: Length of the cylindrical section (x4)

        Returns:
            TensorLike: The objective function values with penalty terms.
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        f = (
            0.6224 * x1 * x3 * x4 +
            1.7781 * x2 * x3**2 +
            3.1661 * x1**2 * x4 +
            19.84 * x1**2 * x3
        )

        g1 = -x1 + 0.0193 * x3
        g2 = -x2 + 0.00954 * x3
        g3 = -bm.pi * x3**2 * x4 - (4/3) * bm.pi * x3**3 + 1296000
        g4 = x4 - 240

        penalties = self.penalty(g1) + self.penalty(g2) + self.penalty(g3) + self.penalty(g4)
        return f + penalties