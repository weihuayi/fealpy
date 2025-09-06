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
        self.MaxFes = self.dim * 10000
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
        self.__NF += x.shape[0]
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
    
    def find_memory(self, pop):
        """
        Locates a population individual in the memory archive.

        Parameters:
            pop (Tensor): The candidate individual of shape (1, dim).

        Returns:
            int | None: The index of the matching memory entry if found, otherwise None.
        """
        which1 = bm.where(self.Memory[:, 0] == pop[0, 0])[0]
        i = 1
        while len(which1) > 1:
            which1 = which1[bm.where(self.Memory[which1, i] == pop[0, i])[0]]
            i += 1
        if len(which1) == 1:
            return which1[0]  
        else:
            return None
    
    def save_memory(self, pop, fit):
        """
        Saves a solution and its fitness into the memory archive.

        If the memory is full, the archive is sorted in descending order 
        of fitness to retain higher-quality solutions.

        Parameters:
            pop (Tensor): The solution vector to save.
            fit (float): The corresponding fitness value.
        """
        self.Mw += 1
        if self.Mw > self.Max - 1:
            self.Mw = 0
            sy = bm.argsort(-self.Memory_f)
            self.Memory_f = self.Memory_f[sy]
            self.Memory = self.Memory[sy]
        self.Memory[self.Mw] = bm.copy(pop)
        self.Memory_f[self.Mw] = bm.copy(fit)
    
    def guided_learning_strategy(self, V0, x, fit, Fes):
        """
        Apply guided learning strategy to update solutions.

        This strategy uses a combination of global best information and random
        exploration to guide the search process.

        Parameters:
            V0 (TensorLike): Velocity vector for guidance.
            x (TensorLike): Current population positions.
            fit (TensorLike): Current fitness values.
            Fes (int): Current function evaluation count.

        Returns:
            tuple: Updated positions, fitness values, and evaluation count.
            Returns None if maximum evaluations reached.
        """
        which_dim = V0 > self.A
        x_new = (which_dim * (self.gbest + bm.tan(bm.pi * (bm.random.rand(self.N, self.dim) - 0.5)) * (self.ub - self.lb) / V0) + 
                 ~which_dim * (bm.random.rand(self.N, self.dim) * (self.ub - self.lb)))
        x_new = bm.clip(x_new, self.lb, self.ub)
        fit_new = self.fun(x_new)
        mask = fit_new < fit
        x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
        self.update_gbest(x, fit)
        Fes = Fes + self.N
        if Fes >= self.MaxFes:
            return
        return x, fit, Fes
    
    def thinking_innovation_strategy(self, pop):
        """
        Applies the Thinking Innovation Strategy (TIS) to update a population in a metaheuristic algorithm
        based on Depth of Knowledge (DOK) and Information Events (IE), improving exploration and exploitation.

        Parameters:
            pop (Tensor): The current individual's position vector in the search space.
            pop_f (float or Tensor): The fitness value of the current individual.

        Attributes used:
            self.fes (int): Current number of function evaluations.
            self.max_fes (int): Maximum allowed function evaluations.
            self.person (Tensor): The stored successful individual used as the Information Event (IE).
            self.person_f (float or Tensor): The fitness value of `self.person`.
            self.fun (Callable): The objective function to evaluate fitness.

        Process:
            1. Recall out-of-bound solutions for `pop` to ensure all variables are within `[lu[0], lu[1]]`.
            2. Compute the Depth of Knowledge (DOK) according to Eq. (1)-(3) from the TIS paper.
            3. Retrieve the Information Event (IE) from `self.person`.
            4. Compute imagination (IM) based on Eq. (4).
            5. Generate a new candidate `pop_new` using Eq. (5).
            6. Recall out-of-bound solutions for `pop_new`.
            7. Evaluate the new candidate and update `pop` or `self.person` based on survival-of-the-fittest.

        Returns:
            Tuple[Tensor, float or Tensor]:
                - Updated individual position (`pop`)
                - Updated fitness value (`pop_f`)
        
        Reference:
        Heming Jia, Xuelian Zhou, Jinrui Zhang.
        Thinking Innovation Strategy (TIS): A novel mechanism for metaheuristic algorithm design and evolutionary update.
        Applied Soft Computing, 2025, 178: 113071.
        """
        pop_f = self.fun(pop)
        C = 0.5
        DOK1 = C + (self.fes/ self.max_fes) ** C
        DOK2 = self.fes ** 10
        DOK = DOK1 + DOK2

        IM = bm.pi * self.person * bm.random.rand(1)

        pop_new = bm.tan(IM - 0.5) * bm.pi + (pop / DOK + self.person)
        pop_new = bm.clip(pop_new, self.lb, self.ub)
        pop_new_f = self.fun(pop_new)
        self.fes = self.fes + 1
        if pop_new_f < pop_f:
            pop = pop_new
            pop_f = pop_new_f
        
        self.person = pop
        self.person_f = pop_f
        
        return pop, pop_f

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
