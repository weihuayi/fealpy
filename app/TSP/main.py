from fealpy.backend import backend_manager as bm
from fealpy.opt import QuantumParticleSwarmOptAlg
from fealpy.opt import SnowmeltOptAlg 
from fealpy.opt import ParticleSwarmOptAlg
from fealpy.opt import CrayfishOptAlg
from fealpy.opt import GreyWolfOptimizer
from fealpy.opt import HoneybadgerOptAlg
from fealpy.opt import AntColonyOptAlg
from TSP_data import TSPdata
from TSP_citys import TravellingSalesmanProblem
from TSP_citys import gbestroute, soler_tsp_with_algorithm, printroute
import time
import matplotlib.pyplot as plt
# bm.set_backend('pytorch')


start_time = time.perf_counter()

qpso_optimizer = QuantumParticleSwarmOptAlg
pso_optimizer = ParticleSwarmOptAlg
sao_optimizer = SnowmeltOptAlg 
coa_optimizer = CrayfishOptAlg
gwo_optimizer = GreyWolfOptimizer
hbo_optimizer = HoneybadgerOptAlg
ant_optimezer = AntColonyOptAlg

num = 7
citys = TSPdata[num]['citys']()
test = TravellingSalesmanProblem(citys)
test.calD()
D = test.D
fobj = lambda x: test.fitness(x)
NP = 100
lb, ub = (0, 1)

optimizers = {  
    'SAO': sao_optimizer, 
    'COA': coa_optimizer,
    'HBO': hbo_optimizer,
    'QPSO': qpso_optimizer,  
    'PSO': pso_optimizer,  
    'GWO': gwo_optimizer, 
    'ACO': ant_optimezer,   
} 

results = {} 
start_time = time.perf_counter()

for algorithm, optimizer in optimizers.items():   
    gbest, gbest_f = soler_tsp_with_algorithm(optimizer, fobj, lb, ub, NP, citys.shape[0], D)  
    route, route_citys = gbestroute(gbest, citys)  
    results[algorithm] = {'route': route, 'route_citys': route_citys, 'gbest_f': gbest_f}  

for algorithm, result in results.items():  
    print(f'The best solution obtained by {algorithm} is:', result['route'])  
    print(f'The best optimal value of the objective function found by {algorithm} is:', result['gbest_f'])  

end_time = time.perf_counter()
running_time = end_time - start_time
print("Running time: ", running_time)

route_citys_all = bm.array([result['route_citys'].tolist() for result in results.values()]) 
alg_name = list(results.keys())  
printroute(route_citys_all, citys, alg_name)