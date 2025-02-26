import time

import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.opt import initialize, opt_alg_options


class PathPlanning:
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
    """

    def __init__(self, MAP, start_point, end_point, method):
        """
        Initializes the PathPlanner with a map, start point, end point, and optimization method.

        Args:
            MAP (ndarray): A 2D array representing the map. Values should be:
                - 0: Free space (traversable).
                - 1: Obstacle (non-traversable).
            start_point (tuple): Coordinates of the start point, e.g., (row, column).
            end_point (tuple): Coordinates of the end point, e.g., (row, column).
            method (class): The optimization algorithm class to use (e.g., `PSO` or `QPSO`).

        Raises:
            ValueError: If the start or end point is invalid (e.g., lies on an obstacle).
        """
        self.MAP = MAP
        self.start_point = start_point
        self.end_point = end_point
        self.method = method
        self.result = None
        self.running_time = None

        # Validate start and end points
        if self.MAP[self.start_point[0]][self.start_point[1]] != 0 or \
           self.MAP[self.end_point[0]][self.end_point[1]] != 0:
            raise ValueError("Error: Invalid start point or end point (lies on an obstacle).")
        
    def solve(self):
        """
        Solves the path planning problem using the specified optimization algorithm.

        Returns:
            dict: A dictionary containing the following keys:
                - "path": The optimal path as a list of coordinates.
                - "distance": The distance of the optimal path.
        """
        

        start_time = time.perf_counter()

        # Initialize the path planning problem
        self.textMAP = PathPlanningProblem(self.MAP, self.start_point, self.end_point)
        self.textMAP.builddata()  # Build the map dictionary

        # Define the fitness function
        fobj = lambda x: self.textMAP.fitness(x)

        # Set algorithm parameters
        N = 20  # Number of particles
        MaxIT = 50  # Maximum iterations
        lb = 0  # Lower bound
        ub = 1  # Upper bound
        dim = self.textMAP.data["landmark"].shape[0]  # Problem dimension

        # Initialize particles and run the optimizer
        xo = initialize(N, dim, ub, lb)
        option = opt_alg_options(xo, fobj, (lb, ub), N, MaxIters=MaxIT)
        self.optimizer = self.method(option)
        self.optimizer.run()

        # Calculate and process the result
        self.result = self.textMAP.calresult(self.optimizer.gbest)
        self.result["path"] = [x for x, y in zip(self.result["path"], self.result["path"][1:] + [None]) if x != y]

        # Record running time
        end_time = time.perf_counter()
        self.running_time = end_time - start_time

        return self.result
    
    def print_results(self):
        """
        Prints the results of the path planning, including the optimal path and running time.
        """
        if self.result is None:
            print("Error: No results available. Please run the `solve` method first.")
        else:
            print('The optimal path distance: ', self.result["fit"])
            print("The optimal path: ", self.result["path"])
            print("Running time: ", self.running_time, "seconds")
            path_x = self.textMAP.data["node"][self.result["path"], 0]
            path_y = self.textMAP.data["node"][self.result["path"], 1]
            print("The opimal path coordinates: ")
            for x, y in zip(path_x, path_y):
                print("({}, {})".format(x, y))
            # self.textMAP.printMAP(self.result, self.running_time)

    def visualize(self):
        """
        Visualizes the map with the optimal path highlighted.
        """
        if self.result is None:
            print("Error: No results available. Please run the `solve` method first.")
        else:
            self.textMAP.printMAP(self.result, self.running_time)


class PathPlanningProblem:
    def __init__(self, MAP, dataS, dataE):
        self.MAP = MAP
        self.dataS = dataS
        self.dataE = dataE
        self.data = {}

    def builddata(self):
        self.data["R"] = 1 # 横平竖直 
        self.data['map'] = self.MAP
        L, _ = label(self.MAP) # 标记连通区

        # 标记障碍体的转角处
        indices = bm.where(bm.array(L) > 0)
        landmark = bm.concatenate([bm.array(L[i]) for i in indices]) 
        self.data['landmark'] = bm.array(landmark)
        
        # 标记可行区域
        node = [[j, i] for i in range(self.MAP.shape[0]) for j in range(self.MAP.shape[1]) if self.MAP[i, j] == 0]
        self. data['node'] = bm.array(node)

        # 计算点与点之间的距离
        self.data['D'] = squareform(pdist(self.data['node']))

        p1, p2 = bm.where(bm.array(self.data['D']) <= bm.array(self.data['R']))
        
        # 创建稀疏矩阵，对应在data['D']的坐标位置
        D = self.data['D'][(p1, p2)].reshape(-1, 1)
        self.data['net'] = sp.csr_matrix((D.flatten(), (p1, p2)), shape = self.data['D'].shape)
        
        
        # 起点和终点
        self.data['noS'] = bm.where((self.data['node'][:, 0] == self.dataS[0]) & (self.data['node'][:, 1] == self.dataS[1]))[0][0]
        self.data['noE'] = bm.where((self.data['node'][:, 0] == self.dataE[0]) & (self.data['node'][:, 1] == self.dataE[1]))[0][0]
        self.data['numLM0'] = 1
        return self.data
    
    def fitness(self, X):
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
    
    def calresult(self, X):
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


    def printMAP(self, result, time):
        b = self.MAP.shape
        self.MAP = 1 - self.MAP

        # 绘制散点
        plt.scatter(self.dataS[0], self.dataS[1], color = 'blue', s = 100)
        plt.scatter(self.dataE[0], self.dataE[1], color = 'green', s = 100)

        # 显示 MAP 图像
        plt.imshow(self.MAP[::1], cmap = 'gray')

        # 生成网格线坐标      
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
        plt.plot([xpath[-1], self.dataE[0]], [ypath[-1], self.dataE[1]], '-', color = 'red')
        
        plt.show()

class PSO:
    def __init__ (self, N, dim, ub, lb, MaxIT, fobj):
        self.N = N
        self.dim = dim
        self.ub = ub
        self.vub = ub * 0.2
        self.lb = lb
        self.vlb = lb * 0.2
        self.MaxIT = MaxIT
        self.fobj = fobj
        self.best = bm.zeros(self.MaxIT)
        self.gbest = bm.zeros((1, self.dim))
        self.gbest_f = 0

    def initialize(self):
        #种群
        a = bm.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb 
        fit = bm.zeros((self.N))
        fit = self.fobj(a)
        #个体最优
        pbest = bm.copy(a)
        pbest_f = bm.copy(fit)
        #全局最优
        gbest_idx = bm.argmin(pbest_f)
        self.gbest_f = pbest_f[gbest_idx]
        self.gbest = pbest[gbest_idx]
        return fit, a, pbest, pbest_f
    
    def updatePGbest(self, fit, x, pbest_f, pbest):
        pbest_f, pbest = (fit, x) if fit < pbest_f else (pbest_f, pbest)
        gbest_f, gbest = (pbest_f, pbest) if pbest_f < self.gbest_f else (self.gbest_f, self.gbest)
        return pbest_f, pbest, gbest_f, gbest

    def cal(self):
        c1 = 2
        c2 = 2
        fit, x, pbest, pbest_f = self.initialize()
        v = bm.zeros([self.N, self.dim])
        for it in range(0, self.MaxIT):
            w = 0.9 - 0.4 * (it / self.MaxIT)
            r1 = bm.random.rand(self.N, 1)
            r2 = bm.random.rand(self.N, 1)
            v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (self.gbest-x)
            v = v + (self.vlb - v) * (v < self.vlb) + (self.vub - v) * (v > self.vub)
            x = x + v
            x = x + (self.lb - x) * (x < self.lb) + (self.ub - x) * (x > self.ub)
            fit = self.fobj(x)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], x, pbest), bm.where(mask, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)
            (self.gbest_f, self.gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < self.gbest_f else (self.gbest_f, self.gbest)

class QPSO(PSO):
    def cal(self):
        fit, a, pbest, pbest_f = self.initialize()
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        #主循环
        for it in range(0, self.MaxIT):
            alpha = 1 - (it + 1) / (2 * self.MaxIT)
            mbest = sum(pbest) / self.N
            phi = bm.random.rand(self.N, self.dim)
            p = phi * pbest + (1 - phi) * self.gbest
            u = bm.random.rand(self.N, self.dim)
            rand = bm.random.rand(self.N, 1)
            a = p + alpha * bm.abs(mbest - a) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))
            a = a + (self.lb - a) * (a < self.lb) + (self.ub - a) * (a > self.ub)
            fit = self.fobj(a)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], a, pbest), bm.where(mask, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)
            (self.gbest_f, self.gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < self.gbest_f else (self.gbest_f, self.gbest)