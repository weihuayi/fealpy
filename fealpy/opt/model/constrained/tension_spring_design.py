from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class TensionSpringDesign:
    """
    A class representing the tension spring design optimization problem.

    This class implements the tension spring design problem, which is a well-known
    constrained optimization benchmark in engineering design. The goal is to minimize
    the weight of a tension spring while satisfying various design constraints.

    Attributes:
        dim (int): The dimensionality of the problem (3 design variables).
        optimal_val (float): The known optimal value for this problem.
    """

    def __init__(self):
        """
        Initializes the TensionSpringDesign problem.

        Sets up the problem dimensions and known optimal value for validation.
        """
        self.dim = 3
        self.optimal_val = 1.2665232788E-02
    
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
            int: The number of design variables (3).
        """
        return self.dim
    
    def get_bounds(self) -> tuple:
        """
        Returns the lower and upper bounds for each design variable.

        Returns:
            tuple: A tuple containing two arrays:
                - Lower bounds: [0.05, 0.25, 2]
                - Upper bounds: [2, 1.3, 15]
        """
        return bm.array([0.05, 0.25, 2]), bm.array([2, 1.3, 15])

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

        Computes the weight of the tension spring design including
        penalty terms for constraint violations.

        Parameters:
            x (TensorLike): Design variable matrix of shape (N, 3), where:
                - x[:, 0]: Wire diameter (d)
                - x[:, 1]: Mean coil diameter (D)
                - x[:, 2]: Number of active coils (N)

        Returns:
            TensorLike: The objective function values with penalty terms.
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        f = (x3 + 2) * x2 * x1**2
        g1 = 1 - ((x2**3) * x3) / (71785 * (x1**4))
        g2 = (4 * (x2**2) - x1 * x2) / (12566 * (x2 * (x1**3) - (x1**4))) + 1 / (5108 * (x1**2)) - 1
        g3 = 1 - (140.45 * x1) / ((x2**2) * x3)
        g4 = ((x1 + x2) / 1.5) - 1
        return f + (self.penalty(g1) + self.penalty(g2) + self.penalty(g3) + self.penalty(g4))