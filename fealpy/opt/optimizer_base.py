from typing import TypedDict, Callable, Tuple, Union, Optional

import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm 
from fealpy.typing import TensorLike, Index, _S
from fealpy import logger
from fealpy.opt.line_search_rules import *

class Optimizer():
    """
    A base class for optimization algorithms.

    This class provides basic functionality for optimization algorithms, including the initialization of 
    the population, objective function evaluations, and tracking of the global best solution. Specific 
    optimization methods (e.g., genetic algorithm, particle swarm optimization) should subclass and 
    implement the `run()` method.

    Attributes:
        options (dict): A dictionary containing the configuration options for the optimizer.
        debug (bool): A flag to enable/disable debugging (default: False).
        __NF (int): A counter for the number of function evaluations.
        x (TensorLike): The current solution/position of the population.
        N (int): The population size.
        MaxIT (int): The maximum number of iterations.
        dim (int): The dimensionality of the solution.
        lb (TensorLike): Lower bound of the solution space.
        ub (TensorLike): Upper bound of the solution space.
        curve (Tensor): Tracks the best fitness value over iterations.
        D_pl (Tensor): Tracks the exploration percentage over iterations.
        D_pt (Tensor): Tracks the exploitation percentage over iterations.
        Div (Tensor): Stores the population diversity for each iteration.
    """
    def __init__(self, options) -> None:
        """
        Initializes the optimizer with configuration options.

        Parameters:
            options (dict): Configuration dictionary that includes initial solution, population size,
                            maximum iterations, dimensionality, domain bounds, and objective function.
        """
        self.options = options 
        self.debug: bool = False
        self.__NF: int = 0
        self.x = options["x0"]
        self.N = options["NP"]
        self.MaxIT = options["MaxIters"]
        self.dim = options["ndim"]
        if options["domain"] is not None:
            self.lb, self.ub = options["domain"]
        self.curve = bm.zeros((self.MaxIT,))
        self.D_pl = bm.zeros((self.MaxIT,))
        self.D_pt = bm.zeros((self.MaxIT,))
        self.Div = bm.zeros((1, self.MaxIT))
        if options["PF"] is not None:
            self.PF = options["PF"]
        if options["NR"] is not None:
            self.Nr = options["NR"]
            self.ngrid = options["ngrid"]
            self.REP = {}  # Repository to store non-dominated solutions
    @property
    def NF(self) -> int:
        """
        Returns the number of times the function value and gradient are calculated.

        Returns:
            int: The function evaluation count.
        """
        return self.__NF

    def fun(self, x: TensorLike):
        """
        Evaluates the objective function and updates the function evaluation counter.

        Parameters:
            x (TensorLike): The input to the objective function.

        Returns:
            The function value (and gradient if applicable for gradient-based methods).
        """
        self.__NF += self.options['NP'] 
        return self.options['objective'](x)

    def update_gbest(self, x, f):
        """
        Updates the global best solution (gbest) based on the current population's fitness.

        Parameters:
            x (TensorLike): The current population's positions (shape: [N, dim]).
            f (TensorLike): Fitness values of the current population (shape: [N,]).

        Logic:
            - Finds the individual with the best (minimum) fitness.
            - Updates the global best solution and fitness if the new best solution is found.
        """
        gbest_idx = bm.argmin(f)
        (self.gbest_f, self.gbest) = (f[gbest_idx], bm.copy(x[gbest_idx])) if f[gbest_idx] < self.gbest_f else (self.gbest_f, self.gbest)

    def run(self):
        """
        The core optimization loop. This method should be overridden in subclasses to implement
        the specific optimization algorithm's iterative process.

        Raises:
            NotImplementedError: This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def D_pl_pt(self, it):
        """
        Calculates the exploration (D_pl) and exploitation (D_pt) percentages.

        Parameters:
            it (int): The current iteration index.

        Logic:
            - D_pl: Measures the population diversity, indicating the exploration degree.
            - D_pt: Measures the exploitation degree, showing the relative change in diversity.
        
        Formulas:
            - D_pl = 100 * current_diversity / max_diversity
            - D_pt = 100 * |current_diversity - max_diversity| / max_diversity

        Safety:
            - A small epsilon is used to avoid division by zero.
        """
        self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(self.x, axis=0) - self.x)) / self.N)
        self.D_pl[it] = 100 * self.Div[0, it] / bm.max(self.Div)
        self.D_pt[it] = 100 * bm.abs(self.Div[0, it] - bm.max(self.Div)) / bm.max(self.Div)

    def print_optimal_result(self):
        """
        Prints the global best solution and its associated fitness value.
        
        Output:
            - Optimal solution: The best solution vector found by the algorithm.
            - Fitness: The fitness value of the best solution.
        """
        print(f"Optimal solution: {self.gbest} \nFitness: {self.gbest_f}")

    def plot_plpt_percen(self):
        """
        Plots the exploration (D_pl) and exploitation (D_pt) percentage curves.

        Visualization:
            - X-axis: Iteration number.
            - Y-axis: Percentage (%).
            - Displays the average exploration and exploitation percentages in the legend.
        """
        plt.plot(self.D_pl, label=(f'Exploration:{round(bm.mean(self.D_pl).item(), 2)}%'))
        plt.plot(self.D_pt, label=(f'Exploitation:{round(bm.mean(self.D_pt).item(), 2)}%'))
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Percentage')
        plt.title("exploration vs exploitation percentage")
        plt.show()  

    def plot_curve(self, label="label"):
        """
        Plots the convergence curve showing the evolution of the best fitness value over iterations.

        Parameters:
            label (str): The label for the plot (default is "label").
        
        Visualization:
            - X-axis: Iteration number.
            - Y-axis: Best fitness value obtained so far (logarithmic scale).
        """
        plt.semilogy(self.curve, label=label)
        plt.xlabel('Iteration')
        plt.ylabel("Best fitness obtained so far")
        plt.title("Convergence curve")
        plt.legend()
        plt.show()

    def cal_spacing(self):
        sorted_id = bm.argsort(self.REP['fit'][:, 0])
        sorted_front = self.REP['fit'][sorted_id]
        distances = bm.linalg.norm(sorted_front[1:] - sorted_front[:-1], axis=1)
        spacing_value = bm.std(distances)
        return spacing_value    
    
    def cal_IGD(self):
        N = self.REP['fit'].shape[0]
        total = 0
        for i in range(N):
            dis = bm.sqrt(bm.sum((self.PF - self.REP['fit'][i])**2, axis = 1))
            min_dis = bm.min(dis)
            total = total + min_dis
        return total / N


def opt_alg_options(
    x0: TensorLike,
    objective,
    domain = None,
    NP: int = 1, # the number of solution points
    NR = None,
    ngrid = None,
    PF = None,
    Preconditioner = None,
    MaxIters: int = 1000,
    MaxFunEvals: int = 10000,
    NormGradTol: float = 1e-6,
    FunValDiff: float = 1e-6,
    StepLength: float = 1.0,
    StepLengthTol: float = 1e-8,
    NumGrad: int = 10,
    LineSearch: Optional[LineSearch] = None,  # 默认值为 None,
    Print: bool = True,
):
    """
    Generate a dictionary of optimization algorithm options.

    Args:
        x0 (TensorLike): Initial solution point(s) for the optimization algorithm.
        objective (callable): Objective function to be minimized.
        domain (optional): Domain constraints for the optimization problem. Defaults to None.
        NP (int, optional): Number of solution points (population size). Defaults to 1.
        Preconditioner (optional): Preconditioner for the optimization algorithm. Defaults to None.
        MaxIters (int, optional): Maximum number of iterations. Defaults to 1000.
        MaxFunEvals (int, optional): Maximum number of function evaluations. Defaults to 10000.
        NormGradTol (float, optional): Tolerance for the norm of the gradient. Defaults to 1e-6.
        FunValDiff (float, optional): Tolerance for the difference in function values. Defaults to 1e-6.
        StepLength (float, optional): Initial step length for the optimization algorithm. Defaults to 1.0.
        StepLengthTol (float, optional): Tolerance for the step length. Defaults to 1e-8.
        NumGrad (int, optional): Number of gradient evaluations. Defaults to 10.
        LineSearch (Optional[LineSearch], optional): Line search method to use. Defaults to None.
        Print (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: A dictionary containing the optimization algorithm options.
    """
    options = {
            "x0": x0,
            "objective": objective,
            "NP": NP,
            "ndim": x0.shape[-1],
            "domain": domain,
            "Preconditioner": Preconditioner,
            "NR": NR,
            "PF": PF,
            "ngrid": ngrid,
            "MaxIters": MaxIters,
            "MaxFunEvals": MaxFunEvals,
            "NormGradTol": NormGradTol,
            "FunValDiff": FunValDiff,
            "StepLength": StepLength,
            "StepLengthTol": StepLengthTol,
            "NumGrad": NumGrad,
            "LineSearch": LineSearch,
            "Print": Print,
            }
    return options 
