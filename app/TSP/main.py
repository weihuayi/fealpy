from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt import QuantumParticleSwarmOptAlg
from fealpy.experimental.opt import SnowmeltOptAlg 
from fealpy.experimental.opt import ParticleSwarmOptAlg
from app.TSP.TSP_data import TSPdata
from app.TSP.TSP_citys import TravellingSalesmanProblem
from app.TSP.TSP_citys import gbest2route, soler_tsp_with_algorithm, printroute
import time
import matplotlib.pyplot as plt

# bm.set_backend('pytorch')

start_time = time.perf_counter()

qpso_optimizer = QuantumParticleSwarmOptAlg
pso_optimizer = ParticleSwarmOptAlg
sao_optimizer = SnowmeltOptAlg 

num = 0
citys = TSPdata[num]['citys']()
test = TravellingSalesmanProblem(citys)
test.calD()
fobj = lambda x: test.fitness(x)
NP = 100
lb, ub = (0, 1)

qpso_gbest, qpso_gbest_f = soler_tsp_with_algorithm(qpso_optimizer, fobj, lb, ub, NP, citys.shape[0])
pso_gbest, pso_gbest_f = soler_tsp_with_algorithm(pso_optimizer, fobj, lb, ub, NP, citys.shape[0])
sao_gbest, sao_gbest_f = soler_tsp_with_algorithm(sao_optimizer, fobj, lb, ub, NP, citys.shape[0])

qpso_route, qpso_route_citys = gbest2route(qpso_gbest, qpso_gbest_f, citys, 'QPSO')
pso_route, pso_route_citys = gbest2route(pso_gbest, pso_gbest_f, citys, 'PSO')
sao_route, sao_route_citys = gbest2route(sao_gbest, sao_gbest_f, citys, 'SAO')

end_time = time.perf_counter()
running_time = end_time - start_time


route_citys = bm.array([qpso_route_citys, 
                  pso_route_citys,
                  sao_route_citys])

alg_name = ['QPSO', 
            'PSO', 
            'SAO']

printroute(route_citys, citys, alg_name)
plt.show()