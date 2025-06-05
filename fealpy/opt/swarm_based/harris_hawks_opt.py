from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

class HarrisHawksOpt(Optimizer):
    """
    A class implementing the Harris Hawks Optimization (HHO) algorithm.
    
    Harris Hawks Optimization algorithm is a nature-inspired algorithm based on the hunting behavior of Harris hawks.
    It uses both exploration (searching for food) and exploitation (chasing prey) strategies.

    Parameters:
        option (dict): A dictionary containing algorithm parameters such as population size, dimensionality, and max iterations.

    Attributes:
        gbest (Tensor): The global best solution found by the algorithm.
        gbest_f (float): The fitness value of the global best solution.
        curve (Tensor): A tensor to store the progress of the best solution over iterations.

    Methods:
        run(): Main method to run the Harris Hawks Optimization algorithm.

    Reference:
        Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen.
        Harris hawks optimization: Algorithm and applications.
        Future Generation Computer Systems, 2019, 97: 849-872.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Harris Hawks Optimization algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings (e.g., population size, dimensionality).
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Harris Hawks Optimization (HHO) algorithm to find the optimal solution.
        
        This method implements the following phases:
        1. **Exploration**: Harris Hawks explore the solution space by searching for prey.
        2. **Exploitation**: Harris Hawks chase the prey and adjust their positions to improve the solution.
        3. **Energy Management**: The hawks' energy influences their movements; energy decreases as they get closer to prey.
        4. **Boundary Handling**: Constraints ensure that solutions remain within defined bounds.
        
        The global best solution and fitness are updated during each iteration.
        """
        # Initial fitness calculation and global best solution
        fit = self.fun(self.x)  # Evaluate fitness for all individuals
        gbest_index = bm.argmin(fit)  # Find the index of the best solution
        self.gbest = self.x[gbest_index]  # Set global best solution
        self.gbest_f = fit[gbest_index]  # Set global best fitness

        # Main loop for iterations
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals
            
            # Calculate the escaping energy based on iteration progress (E1)
            E1 = 2 * (1 - (it / self.MaxIT))
            q = bm.random.rand(self.N, 1)  # Random values for exploration vs exploitation
            escaping_energy = E1 * (2 * bm.random.rand(self.N, 1) - 1)  # Energy for escaping from prey
            x_rand = self.x[bm.random.randint(0, self.N, (self.N,))]  # Random positions from population

            # Exploration and exploitation dynamics
            x_new = ((bm.abs(escaping_energy) >= 1) * ((q < 0.5) * 
                                                       (x_rand - bm.random.rand(self.N, 1) * 
                                                        bm.abs(x_rand - 2 * bm.random.rand(self.N, 1) * self.x)) + 
                                                       (q >= 0.5) * 
                                                       (self.gbest - bm.mean(self.x, axis=0) - bm.random.rand(self.N, 1) * 
                                                        ((self.ub - self.lb) * bm.random.rand(self.N, 1) + self.lb))) + 
                      (bm.abs(escaping_energy) < 1) * (((q >= 0.5)  + (bm.abs(escaping_energy) < 0.5)) * 
                                                       (self.gbest - escaping_energy * bm.abs(self.gbest - self.x)) + 
                                                       ((q >= 0.5)  + (bm.abs(escaping_energy) >= 0.5)) * 
                                                       (self.gbest - self.x - escaping_energy * 
                                                        bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - self.x)) + 
                                                       ((q < 0.5)  + (bm.abs(escaping_energy) >= 0.5)) * 
                                                       (self.gbest - escaping_energy * 
                                                        bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - bm.mean(self.x, axis=0)))))

            # Apply boundary conditions: make sure all new positions are within bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)  # Recalculate fitness for the new positions

            # Compare fitness and update positions accordingly
            x_new = ((fit_new < fit)[:, None] * (x_new) + 
                     (fit_new >= fit)[:, None] * (((q < 0.5)  + (bm.abs(escaping_energy) >= 0.5)) * 
                                                  (self.gbest - escaping_energy * 
                                                   bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - self.x) + 
                                                   bm.random.rand(self.N, self.dim) * levy(self.N, self.dim, 1.5)) + 
                                                  ((q < 0.5)  + (bm.abs(escaping_energy) < 0.5)) * 
                                                   (self.gbest - escaping_energy * 
                                                    bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - bm.mean(self.x, axis=0)) + 
                                                    bm.random.rand(self.N, self.dim) * levy(self.N, self.dim, 1.5))))

            # Apply boundary constraints again
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)  # Recalculate fitness for the new positions

            # Update population based on the new fitness values
            mask = (fit_new < fit)  # Mask to update individuals with better solutions
            self.x = bm.where(mask[:, None], x_new, self.x)
            fit = bm.where(mask, fit_new, fit)

            # Update global best solution
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f  # Store the global best fitness at each iteration
