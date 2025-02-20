from typing import TypedDict, Callable, Tuple, Union, Optional
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm 
from fealpy.typing import TensorLike, Index, _S
from fealpy import logger
from fealpy.opt.line_search_rules import *

def opt_alg_options(
    x0: TensorLike,
    objective,
    domain = None,
    NP: int = 1, # the number of solution points
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

class Optimizer():
    def __init__(self, options) -> None:
        self.options = options 
        self.debug: bool = False
        self.__NF: int = 0
        self.x = options["x0"]
        self.N = options["NP"]
        self.MaxIT = options["MaxIters"]
        self.dim = options["ndim"]
        self.lb, self.ub = options["domain"]
        self.curve = bm.zeros((self.MaxIT,))
        self.D_pl = bm.zeros((self.MaxIT,))
        self.D_pt = bm.zeros((self.MaxIT,))
        self.Div = bm.zeros((1, self.MaxIT))

    @property
    def NF(self) -> int:
        """
        The number of times the function value and gradient are calculated.
        """
        return self.__NF


    def fun(self, x: TensorLike):
        """
        Objective function.
        The counter `self.NF` works automatically when call `fun(x)`.

        Parameters:
            x [TensorLike]: Input of objective function.

        Return:
            The function value, with gradient value for gradient methods.
        """
        self.__NF += self.options['NP'] 
        return self.options['objective'](x)

    def update_gbest(self, x, f):
        """
        Updates the global best solution (gbest) based on the current population.

        Parameters:
        - x (torch.Tensor): The current population positions, shape (N, dim), 
                            where N is the population size and dim is the dimensionality of the solution.
        - f (torch.Tensor): Fitness values of the current population, shape (N,).

        Logic:
        - Finds the index of the individual with the best (minimum) fitness.
        - Updates the global best fitness and position if the new best is better than the current gbest.
        """
        gbest_idx = bm.argmin(f)
        (self.gbest_f, self.gbest) = (f[gbest_idx], bm.copy(x[gbest_idx])) if f[gbest_idx] < self.gbest_f else (self.gbest_f, self.gbest)

    def run(self):
        """
        The main method to run the optimization algorithm (must be implemented in subclasses).

        Instructions:
        - This method should contain the core logic of the optimization algorithm, including:
            - Initialization
            - Iterative process
            - Termination criteria
        - During each iteration, the update_gbest method should be called to update the global best solution.
        
        Note:
        - This is an abstract method and should be overridden in derived classes.
        """
        raise NotImplementedError
    
    def D_pl_pt(self, it):
        """
        Calculates the exploration (D_pl) and exploitation (D_pt) percentages.

        Parameters:
        - it (int): The current iteration index.

        Logic:
        - D_pl: Measures population diversity (exploration degree), 
                indicating the average deviation of individuals from the population mean.
        - D_pt: Measures the exploitation degree, showing the relative change between 
                the current diversity and the maximum diversity observed.
        
        Formulas:
        - D_pl = 100 * current_diversity / max_diversity
        - D_pt = 100 * |current_diversity - max_diversity| / max_diversity

        Safety:
        - Uses a small epsilon to avoid division by zero.
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
        - Shows the average exploration and exploitation percentages in the legend.
        """
        plt.plot(self.D_pl, label=(f'Exploration:{bm.round(bm.mean(self.D_pl), 2)}%'))
        plt.plot(self.D_pt, label=(f'Exploitation:{bm.round(bm.mean(self.D_pt), 2)}%'))
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Percentage')
        plt.title("exploration vs exploitation percentage")
        plt.show()  

    def plot_curve(self, label="label"):
        """
        Plots the convergence curve showing the evolution of the fitness value over iterations.

        Parameters:
        - label (str): The label for the plot (default is "label").

        Visualization:
        - X-axis: Iteration number.
        - Y-axis: Best fitness value obtained so far (logarithmic scale).
        - The curve displays the trend of the global best fitness value over iterations.
        """
        plt.semilogy(self.curve, label=label)
        plt.xlabel('Iteration')
        plt.ylabel("Best fitness obtained so far")
        plt.title("Convergence curve")
        plt.legend()
        plt.show()