from fealpy.backend import backend_manager as bm
# 定义后端
# bm.set_backend('pytorch')
from fealpy.opt import MO_QuantumParticleSwarmOpt, initialize
from fealpy.opt.optimizer_base import opt_alg_options
from fealpy.opt.benchmark.multi_benchmark import multi_benchmark_data as data


# 定义设备
# device = 'cuda'
# bm.set_default_device(device)\

MultiObj = data[0]

NP = 200
NR = 200
lb = MultiObj['lb']
ub = MultiObj['ub']
dim = MultiObj['ndim']
ngrid = 20

x0 = initialize(NP, dim, ub, lb)
options = opt_alg_options(x0, MultiObj['fun'], (lb, ub), NP, NR, PF=MultiObj['PF'], ngrid=ngrid, MaxIters=100)
test = MO_QuantumParticleSwarmOpt(options)
test.run()
print(test.cal_IGD())
print(test.cal_spacing())