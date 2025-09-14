from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class DTLZ1:
    """
    A class representing the DTLZ1 multi-objective benchmark problem.

    DTLZ1 is a scalable multi-objective test problem from the DTLZ test suite.
    It features a linear Pareto-optimal front and is designed to test the ability
    of algorithms to converge to the Pareto front and maintain diversity.

    Attributes:
        n_obj (int): Number of objectives in the optimization problem.
        n_var (int): Number of decision variables.
    """

    def __init__(self, options):
        """
        Initializes the DTLZ1 problem with configuration options.

        Parameters:
            options (dict): Configuration dictionary containing problem parameters.
                Must include the following keys:
                - n_obj (int): Number of objectives (must be at least 2).
                - n_var (int): Number of decision variables (must be at least n_obj).

        Raises:
            KeyError: If required options 'n_obj' or 'n_var' are missing.
            ValueError: If n_obj < 2 or n_var < n_obj.
        """
        self.options = options
        self.n_obj = options['n_obj']
        self.n_var = options['n_var']

    def evaluate(self, x: TensorLike) -> TensorLike:
        """
        Evaluates the DTLZ1 objective functions for given decision variables.

        Computes all objective function values for the DTLZ1 problem, which
        features a linear Pareto-optimal front in the objective space.

        Parameters:
            x (TensorLike): Decision variable matrix of shape (N, n_var), where
                each row represents a candidate solution.

        Returns:
            TensorLike: Objective function matrix of shape (N, n_obj), where
                each row contains the objective values for a candidate solution.
        """
        f = bm.zeros((x.shape[0], self.n_obj))
        xm = self.n_var - self.n_obj + 1
        gx = 100 * (
            xm + bm.sum(
                (x[:, self.n_obj - 1:] - 0.5) ** 2 - 
                bm.cos(20 * bm.pi * (x[:, self.n_obj - 1:] - 0.5)), 
                axis=1
                )
        )
        f[:, 0] = 0.5 * bm.prod(x[:, :self.n_obj-1], axis=1) * (1 + gx)
        for i in range(1, self.n_obj-1):
            f[:, i] = 0.5 * bm.prod(x[:, :self.n_obj-i-1], axis=-1) * (1 - x[:, self.n_obj-i-1]) * (1 + gx)
        f[:, self.n_obj-1] = 0.5 * (1 + gx) * (1 - x[:, 0])
        return f  
    
    def get_bounds(self) -> tuple:
        """
        Returns the lower and upper bounds for all decision variables.

        Returns:
            tuple: A tuple containing two arrays:
                - Lower bounds: Array of zeros with length n_var
                - Upper bounds: Array of ones with length n_var
        """
        return bm.zeros((self.n_var,)), bm.ones((self.n_var))