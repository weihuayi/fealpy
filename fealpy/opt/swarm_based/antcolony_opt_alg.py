import random

from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class AntColonyOptAlg(Optimizer):
    """
    Ant Colony Optimization (ACO) Algorithm for solving combinatorial optimization problems.

    This implementation follows the classical ACO approach to find the optimal path in problems like the Traveling Salesman Problem.

    Parameters:
        option (dict): A dictionary containing algorithm parameters such as population size, max iterations, etc.
        D (Tensor): A distance matrix containing pairwise distances between cities.
    """

    def __init__(self, option, D) -> None:
        """
        Initializes the Ant Colony Optimization algorithm with the provided options.

        Parameters:
            option (dict): Dictionary containing optimizer settings.
            D (Tensor): Distance matrix for the problem (e.g., for TSP).
        """
        super().__init__(option)


    def run(self, params={'alpha':1, 'beta':5, 'rho':0.5, 'Q':1}):
        """
        Runs the Ant Colony Optimization algorithm for optimization.

        The algorithm iterates over several steps where ants construct solutions, update pheromones, and track the best solution.

        Returns:
            gbest (Tensor): The best solution found by the ants.
            gbest_f (float): The fitness (cost) of the best solution.
        """
        # Get the algorithm options
        alpha = params.gat('alpha')
        beta = params.gat('beta')
        rho = params.gat('rho')
        Q = params.gat('Q')
        option = self.options
        x = option["x0"]
        N = option["NP"]  # Number of ants
        T = option["MaxIters"]  # Maximum iterations
        dim = option["ndim"]  # Number of dimensions (cities)
        lb, ub = option["domain"]  # Domain boundaries (not used in this context)

        # Heuristic matrix Eta (1 / distance)
        Eta = 1 / (self.D + 1e-6)

        # Initialize the pheromone matrix and the solution table
        Table = bm.zeros((N, dim), dtype=int)
        Tau = bm.ones((dim, dim), dtype=bm.float64)  # Initial pheromone values
        route_id = bm.arange(dim - 1)  # City indices for routing

        # Initialize global best
        gbest_f = float('inf')
        gbest = None

        # Main loop for the ACO algorithm
        for t in range(T):
            start = [random.randint(0, dim - 1) for _ in range(N)]  # Random start cities for each ant
            
            # Set the starting city for each ant
            Table[:, 0] = bm.array(start)
            citys_index = bm.arange(dim)

            # Construct solution for each ant
            P = bm.zeros((N, dim - 1))
            for j in range(1, dim):
                tabu = Table[:, :j]  # The cities visited by the ants
                w = []
                for i in range(N):
                    # Determine which cities are available for each ant
                    tabu_set = set(tabu[i].tolist())
                    difference_list = list(set(citys_index.tolist()).difference(tabu_set))
                    w.append(difference_list)
                allow = bm.array(w)

                # Compute transition probabilities based on pheromone and heuristic information
                P = (Tau[tabu[:, -1].reshape(-1, 1), allow] ** alpha) * \
                    (Eta[tabu[:, -1].reshape(-1, 1), allow] ** beta)
                P /= P.sum(axis=1, keepdims=True)  # Normalize probabilities
                Pc = bm.cumsum(P, axis=1)  # Cumulative probability
                rand_vals = bm.random.rand(N, 1)  # Random numbers to decide the next city

                # Determine the next city to visit for each ant based on probabilities
                target_index = bm.array([bm.where(row >= rand_val)[0][0] if bm.any(row >= rand_val) else -1
                                         for row, rand_val in zip(Pc, rand_vals.flatten())])
                Table[:, j] = allow[bm.arange(N), target_index]  # Assign the next city

            # Calculate the fitness (total distance) for each ant's solution
            fit = bm.zeros(N)
            fit += bm.sum(self.D[Table[:, route_id], Table[:, route_id + 1]], axis=1)  # Sum of distances between cities
            fit += self.D[Table[:, -1], Table[:, 0]]  # Add the distance to the starting city

            # Update global best solution if a better one is found
            gbest_idx = bm.argmin(fit)
            if fit[gbest_idx] < gbest_f:
                gbest_f = fit[gbest_idx]
                gbest = Table[gbest_idx]

            # Update pheromone matrix based on the ants' solutions
            Delta_Tau = bm.zeros((dim, dim))
            Delta_Tau[Table[:, :-1], Table[:, 1:]] += (Q / fit).reshape(-1, 1)
            Delta_Tau[Table[:, -1], Table[:, 0]] += Q / fit
            Tau = (1 - rho) * Tau + Delta_Tau  # Evaporate pheromones and add new pheromones

        # Return the best solution found and its fitness value
        return gbest, gbest_f
