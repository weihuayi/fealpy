from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class SpeedReducerDesign:
    """
    A class representing the speed reducer design optimization problem.

    This class implements the speed reducer design problem, which is a complex
    constrained optimization benchmark in mechanical engineering design. The goal
    is to minimize the weight of a speed reducer while satisfying various design
    constraints including stress, deflection, and dimensional requirements.

    Attributes:
        dim (int): The dimensionality of the problem (7 design variables).
        optimal_val (float): The known optimal value for this problem.
    """

    def __init__(self):
        """
        Initializes the SpeedReducerDesign problem.

        Sets up the problem dimensions and known optimal value for validation.
        """
        self.dim = 7
        self.optimal_val = 2.9944244658E+03

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
            int: The number of design variables (7).
        """
        return self.dim
    
    def get_bounds(self) -> tuple:
        """
        Returns the lower and upper bounds for each design variable.

        Returns:
            tuple: A tuple containing two arrays:
                - Lower bounds: [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]
                - Upper bounds: [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
        """
        return bm.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]), bm.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
    
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

        Computes the weight of the speed reducer design including
        penalty terms for constraint violations.

        Parameters:
            x (TensorLike): Design variable matrix of shape (N, 7), where:
                - x[:, 0]: Face width (b)
                - x[:, 1]: Module of teeth (m)
                - x[:, 2]: Number of teeth on pinion (z)
                - x[:, 3]: Length of first shaft between bearings (l1)
                - x[:, 4]: Length of second shaft between bearings (l2)
                - x[:, 5]: Diameter of first shaft (d1)
                - x[:, 6]: Diameter of second shaft (d2)

        Returns:
            TensorLike: The objective function values with penalty terms.
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        x7 = x[:, 6]

        f = (
            0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )

        g1 = 27 / (x1 * x2**2 * x3) - 1
        g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
        g3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
        g4 = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
        g5 = (1 / (110 * x6**3)) * bm.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) - 1
        g6 = (1 / (85 * x7**3)) * bm.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) - 1
        g7 = (x2 * x3) / 40 - 1
        g8 = (5 * x2) / x1 - 1
        g9 = x1 / (12 * x2) - 1
        g10 = (1.5 * x6 + 1.9) / x4 - 1
        g11 = (1.1 * x7 + 1.9) / x5 - 1

        penalties = (
            self.penalty(g1) + self.penalty(g2) + self.penalty(g3) + self.penalty(g4) +
            self.penalty(g5) + self.penalty(g6) + self.penalty(g7) + self.penalty(g8) +
            self.penalty(g9) + self.penalty(g10) + self.penalty(g11)
        )

        return f + penalties