from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class InvasiveWeedOpt(Optimizer):
    """
    Invasive Weed Optimization (IWO) algorithm, subclass of Optimizer.

    This class implements the Invasive Weed Optimization (IWO) algorithm, which is a population-based 
    optimization technique inspired by the behavior of invasive weeds. The algorithm simulates the growth 
    and spread of weeds, where each weed represents a potential solution. The algorithm iteratively refines 
    the population based on fitness and diversity measures.

    Parameters:
        option (dict): A dictionary containing the configuration options for the optimizer, such as initial 
                       solution, population size, maximum iterations, dimensionality, bounds, and objective function.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the IWO optimizer by calling the parent class constructor.

        Parameters:
            option (dict): Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self, params={'n_max':100, 's_min':0, 's_max':5, 'n':3, 'sigma_initial':3, 'sigma_final':0.001}):
        """
        Runs the Invasive Weed Optimization (IWO) algorithm.

        This method performs the main optimization loop, updating the population of solutions based on fitness 
        and diversity measures, and tracking the global best solution over iterations.

        Parameters:
            n_max (int): Maximum number of solutions to retain at each iteration (default: 100).
            s_min (float): Minimum seed value (default: 0).
            s_max (float): Maximum seed value (default: 5).
            n (int): Parameter controlling the rate of change for the sigma value (default: 3).
            sigma_initial (float): Initial standard deviation for Gaussian perturbation (default: 3).
            sigma_final (float): Final standard deviation for Gaussian perturbation (default: 0.001).
        """
        # Evaluate the fitness of the initial population
        n_max = params.get('n_max')
        s_min = params.get('s_min')
        s_max = params.get('s_max')
        n = params.get('n')
        sigma_final = params.get('sigma_final')
        sigma_initial = params.get('sigma_initial')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Main optimization loop
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Calculate exploration and exploitation percentages

            # Update sigma based on the iteration number
            sigma = ((self.MaxIT - it) / self.MaxIT) ** n * (sigma_initial - sigma_final) + sigma_final
            
            # Calculate the number of seeds for each solution
            seed_num = bm.floor((s_max - s_min) * (fit - bm.min(fit)) / (bm.max(fit) - bm.min(fit)) + s_min)
            seed = bm.zeros((int(bm.sum(seed_num)), self.dim))
            current_seed_index = 0
            
            # Generate new seeds (perturbed solutions)
            for i in range(self.x.shape[0]):
                if seed_num[i] > 0:
                    seed[current_seed_index:current_seed_index + int(seed_num[i])] = self.x[i] + sigma * bm.random.randn(int(seed_num[i]), self.dim)
                    current_seed_index += int(seed_num[i])

            # Ensure that seeds stay within the domain bounds
            seed = seed + (self.lb - seed) * (seed < self.lb) + (self.ub - seed) * (seed > self.ub)

            # Evaluate the fitness of the new seeds
            fit_need = self.fun(seed)

            # Combine the original population and the new seeds
            self.x = bm.concatenate([self.x, seed], axis=0)
            fit = bm.concatenate([fit, fit_need])

            # Sort the population by fitness (ascending)
            index = bm.argsort(fit)
            fit = fit[index]
            self.x = self.x[index]

            # Retain only the best n_max solutions
            if self.x.shape[0] > n_max:
                fit = fit[:n_max]
                self.x = self.x[:n_max]

            # Update the global best solution if necessary
            self.update_gbest(self.x, fit)

            # Store the best fitness value for this iteration
            self.curve[it] = self.gbest_f
