from fealpy.opt import QuantumParticleSwarmOpt, initialize
from fealpy.opt.optimizer_base import opt_alg_options
from fealpy.opt.benchmark.constrained_benchmark import constrained_benchmark_data as data

num = 0
fobj = data[num]["objective"]
lb = data[num]["lb"]
ub = data[num]["ub"]
dim = data[num]["ndim"]
N = 100
T = 10000

x0 = initialize(N, dim, ub, lb)
option = opt_alg_options(x0, fobj, (lb, ub), N, MaxIters=T)
optimizer = QuantumParticleSwarmOpt(option)
optimizer.run()
optimizer.print_optimal_result()
optimizer.plot_curve("Grey Wolf Optimizer")
optimizer.plot_plpt_percen()