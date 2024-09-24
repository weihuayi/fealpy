from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt import QuantumParticleSwarmOptAlg
from fealpy.experimental.opt import SnowmeltOptAlg 
from fealpy.experimental.opt import ParticleSwarmOptAlg
from app.TSP.TSP_data import TSPdata
from app.TSP.TSP_citys import TravellingSalesmanProblem
from app.TSP.TSP_citys import gbest2route, soler_tsp_with_algorithm, calD, printroute
import time
# bm.set_backend('pytorch')

start_time = time.perf_counter()

qpso_optimizer = QuantumParticleSwarmOptAlg
pso_optimizer = ParticleSwarmOptAlg
sao_optimizer = SnowmeltOptAlg 

num = 0
citys = TSPdata[num]['citys']()
D = calD(citys)
test = TravellingSalesmanProblem(citys, D)
fobj = lambda x: test.fitness(x)
NP = 100
lb, ub = (0, 1)

qpso_gbest, qpso_gbest_f = soler_tsp_with_algorithm(qpso_optimizer, fobj, lb, ub, NP, citys.shape[0])
pso_gbest, pso_gbest_f = soler_tsp_with_algorithm(pso_optimizer, fobj, lb, ub, NP, citys.shape[0])
sao_gbest, sao_gbest_f = soler_tsp_with_algorithm(sao_optimizer, fobj, lb, ub, NP, citys.shape[0])

qpso_route, qpso_route_citys = gbest2route(qpso_gbest, citys)
pso_route, pso_route_citys = gbest2route(pso_gbest, citys)
sao_route, sao_route_citys = gbest2route(sao_gbest, citys)

end_time = time.perf_counter()
running_time = end_time - start_time
print('The best solution obtained by QPSO is:', qpso_route)
print('The best optimal value of the objective function found by QPSO is:', qpso_gbest_f)
print('The best solution obtained by PSO is:', pso_route)
print('The best optimal value of the objective function found by PSO is:', pso_gbest_f)
print('The best solution obtained by PSO is:', sao_route)
print('The best optimal value of the objective function found by PSO is:', sao_gbest_f)
print("Running time: ", running_time)

route_citys = bm.array([qpso_route_citys, 
                  pso_route_citys,
                  sao_route_citys])

alg_name = ['QPSO', 
            'PSO', 
            'SAO']

printroute(route_citys, citys, alg_name)

