import matplotlib.pyplot as plt

from ..backend import backend_manager as bm
from ..opt import *
from .travelling_salesman_prob import TravellingSalesmanProb

class MultipleTravelingSalesmanProb(TravellingSalesmanProb):
    """
    A class for solving the Multiple Traveling Salesman Problem (mTSP) using a two-level optimization approach.
    Inherits from TravellingSalesmanProb and extends functionality for multiple UAV path planning.

    The upper-level optimization assigns targets to UAVs, while the lower-level optimizes individual UAV paths.
    Uses bio-inspired optimization algorithms for both levels.

    Parameters:
        uav_num (int): Number of UAVs (salesmen) in the problem.
        pos (array): Node positions including depot (last position is depot).
        up_opt_dict (dict): Dictionary containing upper-level optimization configuration with keys:
            - 'opt_alg': Optimization algorithm class
            - 'NP': Population size
            - 'MaxIT': Maximum iterations
        down_opt_dict (dict): Dictionary containing lower-level optimization configuration with keys:
            - 'opt_alg': Optimization algorithm class
            - 'NP': Population size
            - 'MaxIT': Maximum iterations

    Attributes:
        uav_num (int): Stores the number of UAVs.
        up_alg (Optimizer): Upper-level optimization algorithm.
        up_N (int): Upper-level population size.
        up_T (int): Upper-level max iterations.
        down_alg (Optimizer): Lower-level optimization algorithm.
        down_N (int): Lower-level population size.
        down_T (int): Lower-level max iterations.
        D_arrays (list): Stores distance matrices for each UAV's targets.
        salesman_path (list): Stores paths for each UAV in the population.
        route (list): Stores final optimized routes for each UAV.
    """

    def __init__(self, uav_num, pos, up_opt_dict, down_opt_dict):
        super().__init__(pos)
        self.uav_num = uav_num
        self.up_alg = up_opt_dict['opt_alg']
        self.up_N = up_opt_dict['NP']
        self.up_T = up_opt_dict['MaxIT']
        self.down_alg = down_opt_dict['opt_alg']
        self.down_N = down_opt_dict['NP']
        self.down_T = down_opt_dict['MaxIT']

    def up_fobj(self, x):
        """
        Upper-level objective function. Creates paths from solution encoding and evaluates fitness.

        Parameters:
            x (Tensor): Solution encoding representing target assignments.

        Returns:
            float: Maximum path length among all UAVs (minimization objective).
        """
        self.create_paths(x)
        fit = self.opt_paths()
        return fit

    def opt_paths(self):
        """
        Optimizes paths for all UAVs using lower-level optimization.

        Returns:
            Tensor: Fitness values (path lengths) for each solution in population.
        """
        fit = bm.zeros((self.up_N,))
        NP = self.down_N
        MaxIT = self.down_T
        lb = 0
        ub = 1  
        for i in range(self.up_N):
            f = bm.zeros((self.uav_num, 1))
            for j in range(self.uav_num):
                fobj = lambda x: self.down_fobj(x, self.D_arrays[i][j])
                dim = self.D_arrays[i][j].shape[0]
                if dim == 0:
                    continue
                x0 = initialize(NP, dim, ub, lb)
                option = opt_alg_options(x0, fobj, (lb, ub), NP, MaxIters=MaxIT)
                self.down_optimizer = self.down_alg(option)
                self.down_optimizer.run()
                f[j] = self.down_optimizer.gbest_f
            fit[i] = bm.max(f)
        return fit

    def down_fobj(self, x, D):
        """
        Lower-level objective function evaluating path length for a single UAV.

        Parameters:
            x (Tensor): Solution encoding representing path order.
            D (Tensor): Distance matrix for current UAV's targets.

        Returns:
            float: Total path length for the given solution.
        """
        return self.path_len(x, D)

    def create_paths(self, x):
        """
        Creates initial paths by assigning targets to UAVs based on solution encoding.

        Parameters:
            x (Tensor): Solution encoding representing target assignments.
        """
        self.D_arrays = []
        self.salesman_path = []
        divs = self.set_divs()
        assigned = bm.searchsorted(divs, x)
        center_idx = self.pos.shape[0]

        for k in range(x.shape[0]):
            path = []
            D_array = []
            for i in range(self.uav_num):
                mask = assigned[k] == i
                rr = bm.nonzero(mask)[0]
                route = bm.concatenate((rr, bm.array([center_idx - 1])))
                path.append(route)
                D_array.append(self.construct_D(route))
            self.salesman_path.append(path)
            self.D_arrays.append(D_array)

    def construct_D(self, r):
        """
        Constructs distance matrix for a given route.

        Parameters:
            r (Tensor): Indices of nodes in the route.

        Returns:
            Tensor: Distance matrix for the route.
        """
        return self.D[r[:, None], r[None, :]]

    def set_divs(self):
        """
        Sets division points for target assignment to UAVs.

        Returns:
            Tensor: Division points in [0,1] range for assignment.
        """
        divs = (bm.arange(self.uav_num) + 1) / self.uav_num
        return divs

    def solver(self):
        """
        Main solver method that runs the two-level optimization.
        """
        fobj = lambda x: self.up_fobj(x)
        lb = 0
        ub = 1
        NP = self.up_N
        MaxIT = self.up_T
        dim = self.pos.shape[0] - 1
        x0 = initialize(NP, dim, ub, lb)
        option = opt_alg_options(x0, fobj, (lb, ub), NP, MaxIters=MaxIT)
        self.up_optimizer = self.up_alg(option)
        self.up_optimizer.run()

    def visualization(self):
        """
        Visualizes the final UAV routes using matplotlib.
        """
        plt.scatter(self.pos[:, 0], self.pos[:, 1], color='red')
        for i, r in enumerate(self.route):
            r_closed = bm.concatenate([r, r[:1]])
            plt.plot(self.pos[r_closed, 0], self.pos[r_closed, 1], label=f'UAV-{i}')
        plt.legend()
        plt.title("UAV Routing Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    def output_route(self, x):
        """
        Outputs the final optimized routes for each UAV.

        Parameters:
            x (Tensor): Final solution encoding from upper-level optimization.
        """
        self.route = [] 
        self.distence = bm.zeros((self.uav_num, 1))
        divs = self.set_divs()
        assigned = bm.searchsorted(divs, x)
        for i in range(self.uav_num):
            mask = assigned == i
            rr = bm.nonzero(mask)[0]
            path = bm.concatenate((rr, bm.array([self.D.shape[0]-1])))
            D = self.construct_D(path)
            dim = D.shape[0]
            fobj = lambda x: self.down_fobj(x, D)
            x0 = initialize(self.down_N, dim, 1, 0)
            option = opt_alg_options(x0, fobj, (0, 1), self.down_N, MaxIters=self.down_T)
            optimizer = self.down_alg(option)
            optimizer.run()
            best = bm.argsort(optimizer.gbest)
            self.route.append(path[best])
            self.distence[i] = optimizer.gbest_f
            print('UAV: ', i)
            print('Route: ', self.route[i])
            print('Dis: ', self.distence[i])

if __name__ == "__main__":
    pos = bm.array([
        [1150, 1760],
        [630, 1660],
        [40, 2090],
        [750, 1100],
        [750, 2030],
        [1030, 2070],
        [1650, 650],
        [1490, 1630],
        [790, 2260],
        [710, 1310],
        [840, 550],
        [1170, 2300],
        [970, 1340],
        [510, 700],
        [750, 900],
        [1280, 1200],
        [230, 590],
        [460, 860],
        [1040, 950],
        [590, 1390],
        [830, 1770],
        [490, 500],
        [1840, 1240],
        [1260, 1500],
        [1280, 790],
        [490, 2130],
        [1460, 1420],
        [1260, 1910],
        [360, 1980],
        [420, 1930]
    ])
    # 城市坐标，其中仓库坐标在最后
    data = bm.concatenate([pos, bm.mean(pos, axis=0)[None, :]])

    up_opt_dict = {
        'opt_alg': QuantumParticleSwarmOpt,
        'NP': 10,
        'MaxIT': 100
    }
    down_opt_dict = {
        'opt_alg': SnowAblationOpt,
        'NP': 10,
        'MaxIT': 100
    }
    # x = bm.array([
    #     0.795779845877519, 0.385844036832549, 0.414651548868334, 0.323053573038183,
    #     0.373677034942638, 0.417871127159426, 0.133358358046610, 0.893566951350942,
    #     0.338608928575216, 0.271748915588681, 0.805797150823545, 0.931355611047065,
    #     0.174806841621164, 0.555271023988564, 0.770588324316173, 0.0176275299639104,
    #     0.589370283601341, 0.580994954907539, 0.883991924568948, 0.116954683199943,
    #     0.157348128915635, 0.551508590765413, 0.535816784345384, 0.294497775198603,
    #     0.0611877302386875, 0.274080690797622, 0.762050220557501, 0.862193919973734,
    #     0.149771051886312, 0.0678699776102234
    # ])
    text = MultipleTravelingSalesmanProb(4, data, up_opt_dict, down_opt_dict)
    text.solver()
    text.output_route(text.up_optimizer.gbest)
    text.visualization()