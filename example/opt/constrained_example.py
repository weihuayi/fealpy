from fealpy.opt import GreyWolfOpt, initialize
from fealpy.opt.optimizer_base import opt_alg_options

def penalty(value):
    return 0 + ((0 < value) * (value < 1)) * value + (value >= 1) * (value ** 2)

def fobj(x):
    def f(x):
        return (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2
    def g1(x):
        return (x[:, 0] - 1) ** 3 - x[:, 1] + 1
    def g2(x):
        return x[:, 0] + x[:, 1] - 2
    return f(x) + penalty(g1(x)) + penalty(g2(x))

lb = [-1.5, -0.5]
ub = [1.5, 2.5]
x0 = initialize(50, 2, ub, lb)
option = opt_alg_options(x0, fobj, (lb, ub), 50, MaxIters=100)
optimizer = GreyWolfOpt(option)
optimizer.run()
optimizer.print_optimal_result()
optimizer.plot_curve("Grey Wolf Optimizer")
optimizer.plot_plpt_percen()