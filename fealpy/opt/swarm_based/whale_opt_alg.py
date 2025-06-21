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
