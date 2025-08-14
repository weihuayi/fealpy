from fealpy.opt import CEC2017, CEC2020, CEC2022
from fealpy.opt import *
from fealpy.opt.optimizer_base import opt_alg_options


func_num = 2
dim = 2
cec = CEC2020(func_num=func_num, dim=dim)
lb = cec.lb
ub = cec.ub
NP = 100

fobj = lambda x: cec.evaluate(x)
x0 = initialize(NP, dim, ub, lb, method=None)
option = opt_alg_options(x0, fobj, (lb, ub), NP)
optimizer = TIS_MarinePredatorsAlg(option)
optimizer.run()
optimizer.print_optimal_result()

