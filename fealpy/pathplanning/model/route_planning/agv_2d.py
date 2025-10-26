import time

import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.opt import initialize, opt_alg_options


class AGV2D:
    """
    A class for solving path planning problems using metaheuristic optimization algorithms.

    This class encapsulates the functionality for finding the optimal path between a start
    point and an end point on a given map. The map is represented as a 2D array, where
    obstacles are marked with `1` and free spaces with `0`.

    Attributes:
        MAP (ndarray): A 2D array representing the map.
        start_point (tuple): Coordinates of the start point, e.g., (row, column).
        end_point (tuple): Coordinates of the end point, e.g., (row, column).
        method (class): The optimization algorithm class to use (e.g., `PSO` or `QPSO`).
        result (dict): The result of the path planning, including the optimal path and distance.
        running_time (float): The time taken to solve the path planning problem.
        data (dict): Internal data structure for storing intermediate computation results.
    """

    def __init__(self, options):
        """
        Initializes the PathPlanner with a map, start point, end point, and optimization method.

        Parameters:
            MAP (ndarray): A 2D array representing the map. Values should be:
                - 0: Free space (traversable).
                - 1: Obstacle (non-traversable).
            start_point (tuple): Coordinates of the start point, e.g., (row, column).
            end_point (tuple): Coordinates of the end point, e.g., (row, column).
            method (class): The optimization algorithm class to use (e.g., `PSO` or `QPSO`).

        Raises:
            ValueError: If the start or end point is invalid (e.g., lies on an obstacle).
        """
        self.options = options
        self.MAP = options['MAP']
        self.start_point = options['start_point']
        self.end_point = options['end_point']
        self.method = options['opt_alg']
        self.result = None
        self.running_time = None
        self.data = {}

        # Validate start and end points
        if self.MAP[self.start_point[0]][self.start_point[1]] != 0 or \
           self.MAP[self.end_point[0]][self.end_point[1]] != 0:
            raise ValueError("Error: Invalid start point or end point (lies on an obstacle).")
        
    def builddata(self) -> dict:
        """
        Build the internal data structure for path planning computation.

        Constructs the graph representation of the map, identifies obstacles and free spaces,
        computes distances between nodes, and sets up the network for path finding.

        Returns:
            dict: The constructed data dictionary containing:
                - 'R': Movement radius constraint
                - 'map': Original map array
                - 'landmark': Obstacle landmark positions
                - 'node': Coordinates of traversable nodes
                - 'D': Distance matrix between nodes
                - 'net': Sparse network graph representation
                - 'noS': Index of start point in node list
                - 'noE': Index of end point in node list
                - 'numLM0': Number of landmark points
        """
        self.data["R"] = 1  # Horizontal and vertical movement only
        self.data['map'] = self.MAP
        L, _ = label(self.MAP)  # Label connected regions

        # Mark obstacle corner points
        indices = bm.where(bm.array(L) > 0)
        landmark = bm.concatenate([bm.array(L[i]) for i in indices]) 
        self.data['landmark'] = bm.array(landmark)
        
        # Mark traversable areas
        node = [[j, i] for i in range(self.MAP.shape[0]) for j in range(self.MAP.shape[1]) if self.MAP[i, j] == 0]
        self.data['node'] = bm.array(node)

        # Calculate distances between points
        self.data['D'] = squareform(pdist(self.data['node']))

        p1, p2 = bm.where(bm.array(self.data['D']) <= bm.array(self.data['R']))
        
        # Create sparse matrix corresponding to coordinates in data['D']
        D = self.data['D'][(p1, p2)].reshape(-1, 1)
        self.data['net'] = sp.csr_matrix((D.flatten(), (p1, p2)), shape = self.data['D'].shape)
        
        
        # Start and end points
        self.data['noS'] = bm.where((self.data['node'][:, 0] == self.start_point[0]) & (self.data['node'][:, 1] == self.start_point[1]))[0][0]
        self.data['noE'] = bm.where((self.data['node'][:, 0] == self.end_point[0]) & (self.data['node'][:, 1] == self.end_point[1]))[0][0]
        self.data['numLM0'] = 1
        return self.data
    
    def solver(self) -> dict:
        """
        Solves the path planning problem using the specified optimization algorithm.

        Executes the optimization process to find the shortest path from start to end point
        while avoiding obstacles. Uses metaheuristic algorithms to explore the solution space.

        Returns:
            dict: A dictionary containing the following keys:
                - "path": The optimal path as a list of node indices.
                - "fit": The total distance of the optimal path.
        """
        start_time = time.perf_counter()

        # Initialize the path planning problem
        self.builddata()  # Build the map dictionary

        # Define the fitness function
        fobj = lambda x: self.cost_function(x)

        # Set algorithm parameters
        N = 20  # Number of particles
        MaxIT = 50  # Maximum iterations
        lb = 0  # Lower bound
        ub = 1  # Upper bound
        dim = self.data["landmark"].shape[0]  # Problem dimension

        # Initialize particles and run the optimizer
        xo = initialize(N, dim, ub, lb)
        option = opt_alg_options(xo, fobj, (lb, ub), N, MaxIters=MaxIT)
        self.optimizer = self.method(option)
        self.optimizer.run()

        # Calculate and process the result
        self.result = self.calresult(self.optimizer.gbest)
        self.result["path"] = [x for x, y in zip(self.result["path"], self.result["path"][1:] + [None]) if x != y]

        # Record running time
        end_time = time.perf_counter()
        self.running_time = end_time - start_time

        return self.result
    
    def print_results(self) -> None:
        """
        Prints the results of the path planning, including the optimal path and running time.

        Displays the optimal path distance, the sequence of nodes in the path, running time,
        and the coordinates of each point in the optimal path.
        """
        if self.result is None:
            print("Error: No results available. Please run the `solve` method first.")
        else:
            print('The optimal path distance: ', self.result["fit"])
            print("The optimal path: ", self.result["path"])
            print("Running time: ", self.running_time, "seconds")
            path_x = self.data["node"][self.result["path"], 0]
            path_y = self.data["node"][self.result["path"], 1]
            print("The optimal path coordinates: ")
            for x, y in zip(path_x, path_y):
                print("({}, {})".format(x, y))

    def cost_function(self, X: TensorLike) -> TensorLike:
        """
        Calculate the cost (total distance) for given solution candidates.

        Evaluates the fitness of potential solutions by computing the shortest path
        through the specified landmark points using network graph algorithms.

        Parameters:
            X (TensorLike): Solution candidate matrix of shape (N, dim).

        Returns:
            TensorLike: Fitness values for each solution candidate, shape (N,).
        """
        sorted_numbers = bm.argsort(X)
        G = nx.DiGraph(self.data["net"])
        sorted_numbers_flat = sorted_numbers[:, :self.data['numLM0']]
        noS = bm.full((X.shape[0],), self.data['noS']).reshape(-1, 1)
        noE = bm.full((X.shape[0],), self.data['noE']).reshape(-1, 1)
        path0 = bm.concatenate((noS, sorted_numbers_flat, noE), axis=1)
        distances = bm.zeros((X.shape[0], path0.shape[1] - 1))
        for j in range(0, X.shape[0]):   
            distance = [nx.shortest_path_length(G, source=int(path0[j][i]), target=int(path0[j][i + 1]), weight=None) for i in range(path0.shape[1] - 1)]
            distances[j, :] = bm.tensor(distance)
        fit = bm.sum(distances, axis=1)
        return fit
    
    def calresult(self, X: TensorLike) -> dict:
        """
        Calculate the detailed result for the best solution.

        Computes the complete path and total distance for the optimal solution
        found by the optimization algorithm.

        Parameters:
            X (TensorLike): The best solution found by the optimizer.

        Returns:
            dict: Result dictionary containing:
                - "fit": Total path distance
                - "path": Complete path as list of node indices
        """
        result = {}
        sorted_numbers = bm.argsort(X)
        G = nx.DiGraph(self.data["net"])
        distances = []
        paths = []
        sorted_numbers_flat = sorted_numbers[0:self.data['numLM0']]
        sorted_numbers_flat = [element for element in sorted_numbers_flat]
        path0 = [self.data['noS']] + sorted_numbers_flat + [self.data['noE']]
        for i in range(0, len(path0) - 1):
            source = path0[i]
            target = path0[i + 1]
            source = int(source)
            target = int(target)
            path = nx.shortest_path(G, source = source, target = target)
            distance = nx.shortest_path_length(G, source = source, target = target, weight = None)  
            distances.append(distance)
            paths.append(path)
        fit = sum(distances)
        combined_list = []
        for sublist in paths:
            combined_list.extend(sublist)
        result["fit"] = fit
        result["path"] = combined_list
        return result

    def printMAP(self, result: dict, time: float) -> None:
        """
        Internal method to visualize the map with the optimal path.

        Creates a graphical representation of the map, obstacles, start/end points,
        and the computed optimal path.

        Parameters:
            result (dict): The path planning result containing the optimal path.
            time (float): The computation time for display in the plot title.
        """
        b = self.MAP.shape
        self.MAP = 1 - self.MAP

        # Plot start and end points
        plt.scatter(self.start_point[0], self.start_point[1], color = 'blue', s = 100)
        plt.scatter(self.end_point[0], self.end_point[1], color = 'green', s = 100)

        # Display MAP image
        plt.imshow(self.MAP[::1], cmap = 'gray')

        # Generate grid line coordinates      
        xx = bm.linspace(0,b[1],b[1]) - 0.5
        yy = bm.zeros(b[1]) - 0.5
        for i in range(0, b[0]):
            yy = yy + 1
            plt.plot(xx, yy,'-',color = 'black')
        x = bm.zeros(b[0])-0.5
        y = bm.linspace(0,b[0],b[0])-0.5
        
        for i in range(0, b[1]):
            x = x + 1
            plt.plot(x,y,'-',color = 'black')

        plt.xticks([])
        plt.yticks([])
        plt.title(f'( {round(time, 4)} s) The optimal path from: ')
        xpath = self.data["node"][result["path"], 0]
        ypath = self.data["node"][result["path"], 1]
        plt.plot(xpath, ypath, '-', color = 'red')
        plt.plot([xpath[-1], self.end_point[0]], [ypath[-1], self.end_point[1]], '-', color = 'red')
        
        plt.show()
    
    def visualization(self) -> None:
        """
        Visualizes the map with the optimal path highlighted.

        Displays a graphical representation showing the original map, obstacles,
        start and end points, and the computed optimal path connecting them.
        """
        if self.result is None:
            print("Error: No results available. Please run the `solve` method first.")
        else:
            self.printMAP(self.result, self.running_time)