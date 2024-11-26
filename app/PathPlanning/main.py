import time
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm
from fealpy.opt import *
from fealpy.opt.optimizer_base import opt_alg_options
from TSP_citys import TSP_data as TSPdata
from TSP_citys import TravellingSalesmanProblem
from TSP_citys import gbestroute, soler_tsp_with_algorithm, printroute
from Grid_maps import MAP_data as MAPdata
from Grid_maps import GridProblem

# bm.set_backend('pytorch')


class TSPOptimizerApp:
    def __init__(self, num, NP=130, lb=0, ub=1, MaxIters = 10000):
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
            'SAO': SnowAblationOpt,
            'COA': CrayfishOptAlg,
            'HBA': HoneybadgerAlg,
            'QPSO': QuantumParticleSwarmOpt,
            'PSO': ParticleSwarmOpt,
            'GWO': GreyWolfOptimizer,
            'ACO': AntColonyOptAlg,
            'HO': HippopotamusOptAlg,
            'CPO': CrestedPorcupineOpt,
            'BKA':BlackwingedKiteAlg,
            'BOA':ButterflyOptAlg,
            'CS':CuckooSearchOpt,
            'DE':DifferentialEvolution,
            'ETO':ExponentialTrigonometricOptAlg,
        }
        self.results = {}

    def optimize(self):
        start_time = time.perf_counter()

        for algorithm, optimizer in self.optimizers.items():
            algo_start = time.perf_counter()
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

            # gbest_f = bm.array([gbest_f]) if isinstance(gbest_f, (float, int)) else gbest_f
            route, route_citys = gbestroute(gbest, self.citys)

            algo_end = time.perf_counter()
            algo_runtime = algo_end - algo_start

            self.results[algorithm] = {
                'route': route, 
                'route_citys': route_citys, 
                'gbest_f': gbest_f, 
                'time': algo_runtime}
        
        end_time = time.perf_counter()
        running_time = end_time - start_time

        self.print_results()
        print("Total runtime: ", running_time)
        self.visualize_routes()
    
    def print_results(self):
        for algorithm, result in self.results.items():
            print(f"The best optimal value by {algorithm} is : {float(result['gbest_f']):.4f}  Time: {result['time']:.4f} seconds")

    def visualize_routes(self):
        route_citys_all = bm.array([result['route_citys'].tolist() for result in self.results.values()])
        alg_name = list(self.results.keys())
        printroute(route_citys_all, self.citys, alg_name)


class ShortestPathApp:
    def __init__(self, num, NP=20, lb=0, ub=1, MaxIters = 50):
        self.num = num
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.MaxIters = MaxIters
        self.maps = MAPdata[self.num]['map']()
        self.dim = self.maps.shape[0]^2
        self.start = MAPdata[self.num]['start']
        self.goal = MAPdata[self.num]['goal']
        
        self.optimizers = {
            'SAO': SnowAblationOpt,
            'COA': CrayfishOptAlg,
            'HBA': HoneybadgerAlg,
            'QPSO': QuantumParticleSwarmOpt,
            'PSO': ParticleSwarmOpt,
            'GWO': GreyWolfOptimizer,
            'ACO': AntColonyOptAlg,
            'HO': HippopotamusOptAlg,
            # 'CPO': CrestedPorcupineOpt,
            'BKA': BlackwingedKiteAlg,
            'BOA': ButterflyOptAlg,
            'CS': CuckooSearchOpt,
            'DE': DifferentialEvolution,
            'ETO': ExponentialTrigonometricOptAlg,
        }
        self.results = {}

    def optimize(self):
        
        if self.maps[self.start[0]][self.start[1]] != 0 or self.maps[self.goal[0]][self.goal[1]] != 0: 
            print("Error: Wrong start point or end point")
            return

        textMAP = GridProblem(self.maps, self.start, self.goal)
        textMAP.builddata()
        D = textMAP.data['D']

        fobj = lambda x: textMAP.fitness(x)
        for algo_name, algorithm in self.optimizers.items():
            x0 = self.lb + bm.random.rand(self.NP, self.dim) * (self.ub - self.lb)
            option = opt_alg_options(x0, fobj, (self.lb, self.ub), self.NP, self.MaxIters)
            
            
            if algo_name == 'ACO':
                optimizer = algorithm(option, D)
            else:
                optimizer = algorithm(option)

            start_time = time.perf_counter()
            gbest, gbest_f = optimizer.run()
            end_time = time.perf_counter()
            running_time = end_time - start_time
            print(gbest)
            result = textMAP.calresult(gbest)
            result["path"] = [x for x, y in zip(result["path"], result["path"][1:] + [None]) if x != y]

            self.results[algo_name] = {
                "distance": gbest_f,
                "path": result["path"],
                "time": running_time
            }
            print(f'The best optimal value by {algo_name} is: {float(gbest_f):.4f}  Time: {end_time - start_time:.4f} seconds')

        # textMAP.printMAP(self.results)



if __name__ == "__main__":
    num = 0  

    #Traveling Salesman Problem
    tsp_optimizer = TSPOptimizerApp(num)
    tsp_optimizer.optimize()

    #Shortest path problem
    # short_optimizer = ShortestPathApp(num)
    # short_optimizer.optimize()
