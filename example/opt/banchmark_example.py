import time 

from fealpy.backend import backend_manager as bm
from fealpy.opt import *
from fealpy.opt.optimizer_base import opt_alg_options
from fealpy.opt.benchmark.single_benchmark import single_benchmark_data as iopt_data
# device = 'cpu'

# 定义后端
bm.set_backend('pytorch')
# 定义设备
# device = 'cuda'
# bm.set_default_device(device)

start_time = time.perf_counter()

num = 0
lb, ub = iopt_data[num]['domain']
NP = 100
MaxIters = 1000
dim = iopt_data[num]['ndim']
x0 = initialize(NP, dim, ub, lb, method=None)
option = opt_alg_options(x0, iopt_data[num]['objective'], iopt_data[num]['domain'], NP, MaxIters=MaxIters)
optimizer = DifferentialEvolutionParticleSwarmOpt(option)
optimizer.run()
# optimizer.plot_curve()
# optimizer.plot_plpt_percen()
optimizer.print_optimal_result()
# print("Function times:", optimizer.NF)
end_time = time.perf_counter()
running_time = end_time - start_time
print("Running time :", running_time)