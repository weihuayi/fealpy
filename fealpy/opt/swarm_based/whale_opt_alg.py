from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class WhaleOptAlg(Optimizer):
    """
    A class representing the Whale Optimization Algorithm (WOA), inheriting from the Optimizer class.

    The Whale Optimization Algorithm is a nature-inspired optimization algorithm based on the bubble-net hunting strategy 
    of humpback whales. It utilizes a combination of exploration and exploitation phases to search for optimal solutions 
    by simulating the movement and position adjustment behaviors of humpback whales. The algorithm employs mechanisms 
    such as shrinking encircling and spiral updating to improve solution accuracy.

    Parameters:
        option (dict): A dictionary containing configuration options for the algorithm, including the maximum number of 
                       iterations, lower and upper bounds, population size, and other optimization parameters.

    Attributes:
        gbest (Tensor): The global best solution found so far.
        gbest_f (float): The fitness value corresponding to the global best solution.
        MaxIT (int): The maximum number of iterations for the optimization.
        lb (Tensor): Lower bounds of the solution space.
        ub (Tensor): Upper bounds of the solution space.
        N (int): The size of the population.
        dim (int): The dimensionality of the search space.
        fun (callable): The objective (fitness) function to evaluate solutions.
        x (Tensor): The current population of candidate solutions.
        curve (dict): A dictionary to store the fitness values of the global best solution at each iteration.

    Methods:
        run():
            Executes the Whale Optimization Algorithm. The algorithm updates candidate solutions based on three primary 
            phases: exploration, shrinking encircling, and spiral updating. These phases enable the algorithm to balance 
            between exploring the search space and exploiting the best solutions found.

    Reference:
        Seyedali Mirjalili, Andrew Lewis.
        The Whale Optimization Algorithm.
        Advances in Engineering Software, 2016, 95: 51-67
    """

    def __init__(self, option) -> None:
        """
        Initializes the Whale Optimization Algorithm with the provided options.

        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)


    def run(self):
        """
        Runs the Whale Optimization Algorithm.

        This method performs the iterative optimization process, where the positions of candidate solutions are updated 
        based on the behaviors of humpback whales during bubble-net hunting. The algorithm utilizes exploration, shrinking 
        encircling, and spiral updating mechanisms to iteratively improve the candidate solutions and find the optimal one. 
        The fitness of the solutions is evaluated at each iteration, and the global best solution is updated accordingly.

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Update the positions based on exploration and exploitation phases:
                - Exploration: Search around random solutions using a combination of random movement and position update.
                - Shrinking encircling: Move towards the global best solution, reducing the step size over time.
                - Spiral updating: Update positions in a spiral pattern around the global best solution.
            3. Ensure the new positions remain within the bounds of the solution space.
            4. Evaluate the fitness of the new population.
            5. Update the global best solution if a better one is found.
            6. Track the fitness of the global best solution over iterations.
        """
        # Initial fitness evaluation
        fit = self.fun(self.x)

        # Identify the index of the best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Iterative optimization process
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Calculate coefficients for the update equations
            a = 2 - it * 2 / self.MaxIT
            a2 = -1 - it / self.MaxIT
            A = 2 * a * bm.random.rand(self.N, 1) - a
            C = 2 * bm.random.rand(self.N, 1)
            p = bm.random.rand(self.N, 1)
            l = (a2 - 1) * bm.random.rand(self.N, 1) + 1

            # Randomly select a leader solution for exploration
            x_rand = self.x[rand_leader_index]

            # Exploration phase and shrinking encircling mechanism
            self.x = ((p < 0.5) * ((bm.abs(A) >= 1) * (x_rand - A * bm.abs(C * x_rand - self.x)) +  # Exploration phase
                                    (bm.abs(A) < 1) * (self.gbest - A * bm.abs(C * self.gbest - self.x))) +  # Shrinking encircling mechanism
                     (p >= 0.5) * (bm.abs(self.gbest - self.x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + self.gbest))  # Spiral updating position

            # Ensure the new positions stay within the bounds
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            # Evaluate the fitness of new positions
            fit = self.fun(self.x)

            # Update the global best solution
            self.update_gbest(self.x, fit)

            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f

class ImprovedWhaleOptAlg(Optimizer):
    """
    Improved Whale Optimization Algorithm (IWOA) for optimization problems.
    
    This class implements the Improved Whale Optimization Algorithm (IWOA),
    an enhanced version of the standard Whale Optimization Algorithm (WOA), inspired by the
    bubble-net hunting strategy of humpback whales. The improvement introduces dynamic control
    of the exploration and exploitation phases, which enhances the algorithm's ability to find
    global optima in optimization problems.

    Attributes:
        gbest (bm.Tensor): The global best solution found so far.
        gbest_f (float): The fitness value of the global best solution.
        x (bm.Tensor): The population of candidate solutions.
        lb (bm.Tensor): The lower bounds for the variables.
        ub (bm.Tensor): The upper bounds for the variables.
        MaxIT (int): The maximum number of iterations for the algorithm.
        N (int): The size of the population.
        dim (int): The dimensionality of the problem.
        fun (function): The objective function to minimize.
        curve (bm.Tensor): The fitness values over iterations.

    Reference:
        Seyedali Mirjalili, Andrew Lewis.
        A novel improved whale optimization algorithm to solve numerical optimization and real-world applications.
        Artificial Intelligence Review, 2022, 55: 4605-4716.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Improved Whale Optimization Algorithm with the given options.

        Args:
            option (dict): A dictionary containing parameters for the algorithm,
                           such as population size, bounds, and other configuration options.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Improved Whale Optimization Algorithm to optimize the objective function.
        
        The algorithm iterates through the population, updating the positions of the whales
        based on a combination of exploration and exploitation strategies. These strategies
        are controlled by the coefficients A, C, and p, which evolve over the iterations.
        
        Steps:
            1. Evaluate the fitness of the initial population.
            2. Perform the update of whale positions using dynamic control coefficients.
            3. Update the best solution found so far if a better solution is found.
            4. Track the fitness value of the best solution in each iteration.
            
        The process continues for the maximum number of iterations (MaxIT).
        """
        fit = self.fun(self.x)  # Evaluate the fitness of the current population.
    
        gbest_index = bm.argmin(fit)  # Find the index of the best solution.
        self.gbest = self.x[gbest_index]  # Set the global best solution.
        self.gbest_f = fit[gbest_index]  # Set the fitness of the global best solution.

        # Iterate through each iteration of the algorithm.
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update the position of the whales based on exploration/exploitation.

            # Coefficients for exploration and exploitation.
            A = 2 * (2 - it * 2 / self.MaxIT) * bm.random.rand(self.N, 1) - (2 - it * 2 / self.MaxIT)  # Eq.(3)
            C = 2 * bm.random.rand(self.N, 1)  # Eq.(4)
            p = bm.random.rand(self.N, 1)  # Probability for determining exploration or exploitation.
            l = (-2 - it / self.MaxIT) * bm.random.rand(self.N, 1) + 1  # Eq.(9)

            if it < int(self.MaxIT / 2):  # Exploration phase
                # Randomly select two whales from the population.
                rand1 = self.x[bm.random.randint(0, self.N, (self.N,))]
                rand2 = self.x[bm.random.randint(0, self.N, (self.N,))]
                rand_mean = (rand1 + rand2) / 2  # Compute the mean of the two random whales.
                
                # Select the whale based on the distance to rand1 and rand2.
                mask = bm.linalg.norm(self.x - rand1, axis=1)[:, None] < bm.linalg.norm(self.x - rand2, axis=1)[:, None]
                rand = bm.where(mask, rand2, rand1)
                
                # Update the position of the whale using Eq.(2).
                x_new = ((p < 0.5) * (rand - A * bm.abs(C * rand - self.x)) + 
                         (p >= 0.5) * (rand_mean - A * bm.abs(C * rand_mean - self.x)))
            else:  # Exploitation phase
                # Update the position of the whale using Eq.(6) and Eq.(8).
                x_new = ((p < 0.5) * (self.gbest - A * bm.abs(C * self.gbest - self.x)) +  # Eq.(6)
                         (p >= 0.5) * (bm.abs(self.gbest - self.x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + self.gbest))  # Eq.(8)
            
            # Apply boundary constraints to ensure the new position is within bounds.
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            
            # Evaluate the fitness of the new positions.
            fit_new = self.fun(x_new)
            
            # Update the population if the new solution has a better fitness.
            mask = (fit_new < fit)
            self.x = bm.where(mask[:, None], x_new, self.x)
            fit = bm.where(mask, fit_new, fit)
            
            # Update the global best solution if a better one is found.
            self.update_gbest(self.x, fit)
            
            # Record the best fitness value for this iteration.
            self.curve[it] = self.gbest_f
