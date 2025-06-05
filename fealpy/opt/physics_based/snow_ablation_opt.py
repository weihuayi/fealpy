from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class SnowAblationOpt(Optimizer):
    """
    A class representing the Snow Ablation Optimization (SnowAblationOpt) algorithm, inheriting from the Optimizer class.
    
    This optimization algorithm is designed for solving complex optimization problems by simulating a snow ablation process.
    It uses a combination of elite solutions and random exploration to iteratively improve the solution.
    
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
            Executes the Snow Ablation Optimization algorithm. It iteratively improves the candidate solutions by adjusting 
            their positions based on the current elite solutions and random exploration. The algorithm uses specific equations 
            to guide the exploration process and updates the best solution over multiple iterations.

    Reference
    ~~~~~~~~~
    Lingyun Deng, Sanyang Liu.
    Snow ablation optimizer: A novel metaheuristic technique for numerical optimization and engineering design.
    Expert Systems With Applications, 2023, 225: 120069
    """

    def __init__(self, option) -> None:
        """
        Initializes the SnowAblationOpt algorithm with the provided options.
        
        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)
    
    def run(self):
        """
        Runs the Snow Ablation Optimization algorithm.

        This method performs the iterative optimization process where, in each iteration, the algorithm updates the 
        solutions based on elite solutions and random exploration. The solutions are adjusted using the snow ablation 
        process and specific equations to guide the search. The algorithm iterates until the maximum number of iterations 
        (MaxIT) is reached.

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Sort the population to identify the best solutions.
            3. Update the elite solutions pool with the best, second-best, and third-best solutions, along with the mean 
               of the top half of the population.
            4. Randomly assign solutions to two groups (Na and Nb) and update the solutions in each group using specific 
               equations for snow ablation.
            5. Check for boundary conditions to ensure solutions remain within the defined search space.
            6. Update the global best solution if a better one is found.
            7. Track the global best fitness value over iterations.
        """
        # Initial fitness values for the current population
        Objective_values = self.fun(self.x)

        # Ensure lower and upper bounds are in array form
        if not isinstance(self.lb, list):  
            self.lb = bm.array([self.lb] * self.dim)  
        if not isinstance(self.ub, list):
            self.ub = bm.array([self.ub] * self.dim)

        # Calculate the number of elite solutions (half of the population size)
        N1 = bm.array(int(bm.floor(bm.array(self.N * 0.5))))

        # Sort the fitness values to identify the best solutions
        idx1 = bm.argsort(Objective_values)
        
        # Store the best, second-best, and third-best solutions
        self.gbest = bm.copy(self.x[idx1[0], :])
        self.gbest_f = bm.copy(Objective_values[idx1[0]])

        second_best = bm.copy(self.x[idx1[1], :])
        third_best = bm.copy(self.x[idx1[2], :])

        # Calculate the mean of the top half of the population
        half_best_mean = bm.sum(self.x[idx1[:N1], :], axis=0) / N1
        
        # Create the elite pool (top 3 solutions and half-best mean)
        Elite_pool = bm.concatenate((self.gbest.reshape(self.dim, 1), second_best.reshape(self.dim, 1), 
                                     third_best.reshape(self.dim, 1), half_best_mean.reshape(self.dim, 1)), axis=1)
        Elite_pool = Elite_pool.reshape(4, self.dim)
        
        # Split the population into two groups (Na and Nb)
        index = bm.arange(self.N)
        Na = bm.array(int(self.N / 2))
        Nb = bm.array(int(self.N / 2))

        # Iterative optimization process
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            RB = bm.random.randn(self.N, self.dim)  # Random noise for exploration
            
            # Eq.(9) - Calculate the dynamic snow ablation factor
            DDF = 0.35 + 0.25 * (bm.exp(bm.array(it / self.MaxIT)) - 1) / (bm.exp(bm.array((1)) - 1))
            
            # Eq.(10) - Calculate the snow melt rate
            M = DDF * bm.exp(bm.array(-it / self.MaxIT))

            # Randomly assign solutions to two groups (Na and Nb)
            index1 = bm.unique(bm.random.randint(0, self.N - 1, (Na,)))
            index2 = bm.array(list(set(index.tolist()).difference(index1.tolist())))

            # Update the positions of solutions in group Na
            r1 = bm.random.rand(len(index1), 1)
            k1 = bm.random.randint(0, 3, (len(index1),))
            self.x[index1] = Elite_pool[k1] + RB[index1] * (r1 * (self.gbest - self.x[index1]) + 
                                                             (1 - r1) * (bm.mean(self.x, axis=0) - self.x[index1]))

            Na, Nb = (Na + 1, Nb - 1) if Na < self.N else (Na, Nb)

            # Update the positions of solutions in group Nb
            if Nb >= 1:
                r2 = 2 * bm.random.rand(len(index2), 1) - 1
                self.x[index2] = M * self.gbest + RB[index2] * (r2 * (self.gbest - self.x[index2]) + 
                                                                  (1 - r2) * (bm.mean(self.x, axis=0) - self.x[index2]))

            # Check and adjust the solutions to ensure they stay within the search space
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            
            # Evaluate the fitness of the updated population
            Objective_values = self.fun(self.x)
            
            # Update the global best solution if necessary
            self.update_gbest(self.x, Objective_values)
            
            # Update the elite pool with new best solutions
            idx1 = bm.argsort(Objective_values)
            second_best = bm.copy(self.x[idx1[1], :])
            third_best = bm.copy(self.x[idx1[2], :])
            half_best_mean = bm.sum(self.x[idx1[:N1], :], axis=0) / N1

            Elite_pool[0] = self.gbest
            Elite_pool[1] = second_best
            Elite_pool[2] = third_best
            Elite_pool[3] = half_best_mean

            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f
