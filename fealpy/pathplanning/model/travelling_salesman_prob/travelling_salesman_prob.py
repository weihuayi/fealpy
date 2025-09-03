import os

import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.opt import initialize, opt_alg_options

class TravellingSalesmanProb:
    """
    A class representing the Travelling Salesman Problem (TSP).

    This class handles reading input data (coordinates), computing the pairwise distance 
    matrix, evaluating route lengths, optimizing the TSP tour using a metaheuristic algorithm, 
    and visualizing the resulting route.

    Parameters:
        pos (Tensor): A tensor of shape (n, 2), where each row contains the (x, y) coordinates of a city.

    Attributes:
        pos (Tensor): The coordinates of cities used to define the TSP instance.
        D (Tensor): The precomputed Euclidean distance matrix between all cities.
        optimizer: The optimization algorithm instance used to solve the TSP.
    """

    def __init__(self, options):
        """
        Initializes the TSP instance with a given set of city coordinates.

        Parameters:
            pos (Tensor): A tensor of shape (n, 2), representing the positions of n cities in 2D space.
        """
        self.pos = options['pos']
        self.D = self._compute_distance_matrix(options['pos'])
        self.N = options['NP']
        self.MaxIT = options['MaxIT']
        self.opt_alg = options['opt_alg']

    def read_tsp_file(self, filename):
        """
        Reads a `.tsp` file containing city coordinates in TSPLIB format and returns them as a tensor.

        The function searches for the 'NODE_COORD_SECTION' header and parses each subsequent line as a city coordinate 
        until it encounters 'EOF'. The resulting coordinates are returned as a 2D tensor.

        Parameters:
            filename (str): The relative file path of the `.tsp` file to load.

        Returns:
            Tensor: A tensor of shape (n, 2), where each row contains the (x, y) coordinate of a city.
        """
        coords = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            start = False
            for line in lines:
                line = line.strip()
                if line == 'NODE_COORD_SECTION':
                    start = True
                    continue
                if line == 'EOF':
                    break
                if start:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = float(parts[1]), float(parts[2])
                        coords.append((x, y))
        return bm.array(coords)

    def _compute_distance_matrix(self, pos):
        """
        Computes the pairwise Euclidean distance matrix between all city coordinates.

        Given an array of city coordinates, this method computes the L2 norm (Euclidean distance) 
        between every pair of cities and returns the resulting matrix.

        Parameters:
            pos (Tensor): A tensor of shape (n, 2) representing the coordinates of cities.

        Returns:
            Tensor: A distance matrix of shape (n, n), where entry (i, j) is the distance between city i and city j.
        """
        diff = pos[:, None, :] - pos[None, :, :]
        return bm.linalg.norm(diff, axis=2)

    def cost_function(self, sol, D):
        """
        Calculates the total route length of each solution in the population based on a given distance matrix.

        The input `sol` is a batch of candidate solutions, and each solution is a permutation of cities.
        The total travel distance for each route is computed and returned.

        Parameters:
            sol (Tensor): A tensor of shape (N, n), where N is the number of solutions and each row 
                          represents a permutation of city indices.
            D (Tensor): A precomputed distance matrix of shape (n, n).

        Returns:
            Tensor: A tensor of shape (N,) where each value is the total route length of a solution.
        """
        x = bm.argsort(sol, axis=-1)
        n = x.shape[1]
        length = D[x[:, n - 1], x[:, 0]]
        for i in range(1, n):
            length = length + D[x[:, i - 1], x[:, i]]
        return length

    def solver(self):
        """
        Runs the specified optimization algorithm to solve the TSP.

        This method initializes a population of random solutions, defines the objective 
        function (route length), configures the algorithm options, and invokes the optimizer.

        Parameters:
            N (int): The number of individuals (solutions) in the population.
            MaxIT (int): The maximum number of optimization iterations.
            opt_alg (Callable): A class or function implementing the optimization algorithm.
                                Must accept an `option` object and support a `.run()` method.

        Side Effects:
            Initializes and stores the optimizer instance in `self.optimizer`, which contains the best solution.
        """
        dim = self.D.shape[0]
        fobj = lambda x: self.cost_function(x, self.D)
        x0 = initialize(self.N, dim, 1, 0)
        option = opt_alg_options(x0, fobj, (0, 1), self.N, MaxIters=self.MaxIT)
        self.optimizer = self.opt_alg(option)
        self.optimizer.run()
    
    def visualization(self, pos):
        """
        Visualizes the best tour found by the optimization algorithm.

        This method draws the tour using a line plot over the city coordinates and marks each city 
        with a red dot. The route connects all cities in the order of the best solution and loops back 
        to the starting city.

        Parameters:
            pos (Tensor): A tensor of shape (n, 2) representing the coordinates of the cities.

        Side Effects:
            Displays a matplotlib plot showing the best TSP tour.
        """
        best = bm.argsort(self.optimizer.gbest)
        route = bm.concatenate([best, best[:1]])
        route_str = " -> ".join(str(int(i)) for i in route)
        print(f"The optimal route: {route_str}")
        plt.figure(figsize=(8, 6))
        plt.scatter(pos[:, 0], pos[:, 1], color='red')
        for i, (x, y) in enumerate(pos):
            plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom')
        plt.plot(pos[route, 0], pos[route, 1], color='b')
        plt.title("UAV Routing Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

