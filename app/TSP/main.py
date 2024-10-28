import time
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm
from fealpy.opt import *
from fealpy.opt.optimizer_base import opt_alg_options
from TSP_citys import TSP_data as TSPdata
from TSP_citys import TravellingSalesmanProblem
from TSP_citys import gbestroute, soler_tsp_with_algorithm, printroute
# bm.set_backend('pytorch')

class TSPOptimizerApp:
    def __init__(self, num, NP=100, lb=0, ub=1, MaxIters = 10000):
        self.num = num
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.MaxIters = MaxIters
        self.citys = TSPdata[self.num]['citys']()
        self.test = TravellingSalesmanProblem(self.citys)
        self.test.calD()
        self.D = self.test.D
        self.fobj = lambda x: self.test.fitness(x)

        self.optimizers = {
            'SAO': SnowmeltOptAlg,
            'COA': CrayfishOptAlg,
            'HBO': HoneybadgerOptAlg,
            'QPSO': QuantumParticleSwarmOptAlg,
            'PSO': ParticleSwarmOptAlg,
            'GWO': GreyWolfOptimizer,
            'ACO': AntColonyOptAlg,
            'Ho': HippopotamusOptAlg,
        }

        self.results = {}

    def optimize(self):
        start_time = time.perf_counter()

        for algorithm, optimizer in self.optimizers.items():
            if algorithm == 'ACO':
                NP = 10 
                MaxIters = 100 
                x0 = self.lb + bm.random.rand(NP, self.citys.shape[0]) * (self.ub - self.lb)
                option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), NP, MaxIters) 
                optimizer_aco = optimizer(option, self.D)
                gbest, gbest_f = optimizer_aco.run()
            else:
                gbest, gbest_f = soler_tsp_with_algorithm(optimizer, self.fobj, self.lb, self.ub, self.NP, self.citys.shape[0], self.MaxIters)
                gbest = bm.argsort(gbest)

            if isinstance(gbest_f, (float, int)):  
                gbest_f = bm.array([gbest_f])  
            route, route_citys = gbestroute(gbest, self.citys)
            self.results[algorithm] = {'route': route, 'route_citys': route_citys, 'gbest_f': gbest_f}

        end_time = time.perf_counter()
        running_time = end_time - start_time

        self.print_results()
        print("Running time: ", running_time)
        self.visualize_routes()

    def print_results(self):
        for algorithm, result in self.results.items():
            # print(f'The best solution obtained by {algorithm} is:', result['route'])
            print(f'The best optimal value of the objective function found by {algorithm} is:', result['gbest_f'])

    def visualize_routes(self):
        route_citys_all = bm.array([result['route_citys'].tolist() for result in self.results.values()])
        alg_name = list(self.results.keys())
        printroute(route_citys_all, self.citys, alg_name)


if __name__ == "__main__":
    num = 2  # Example city index [0 - 6]
    tsp_optimizer = TSPOptimizerApp(num)
    tsp_optimizer.optimize()
