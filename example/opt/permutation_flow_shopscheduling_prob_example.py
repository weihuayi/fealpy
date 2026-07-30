from fealpy.backend import backend_manager as bm
from fealpy.opt import opt_alg_options, initialize, ExponentialTrigonometricOptAlg
from fealpy.opt.model import OPTModelManager

data = bm.array([
    [54, 51, 44, 66, 18, 100, 51, 88, 41, 92, 12, 98, 11, 16, 13, 83, 17, 59, 44, 31, 44, 9, 90, 29, 81, 53, 40, 66, 16, 78],
    [2, 69, 18, 95, 53, 39, 86, 95, 24, 78, 95, 31, 53, 22, 80, 41, 78, 8, 79, 63, 37, 74, 31, 33, 66, 58, 72, 90, 3, 97],
    [2, 31, 6, 42, 28, 72, 28, 45, 93, 9, 32, 39, 83, 32, 5, 11, 85, 26, 34, 50, 88, 81, 8, 42, 79, 82, 13, 75, 46, 19],
    [75, 100, 6, 1, 61, 19, 56, 83, 33, 47, 60, 91, 29, 21, 57, 64, 87, 77, 85, 39, 43, 55, 79, 54, 36, 91, 79, 45, 2, 22],
    [39, 25, 9, 21, 84, 44, 13, 5, 70, 39, 53, 7, 85, 96, 15, 16, 16, 4, 86, 20, 50, 50, 55, 5, 75, 21, 9, 59, 79, 5],
    [97, 58, 96, 65, 52, 66, 22, 30, 84, 48, 44, 38, 31, 42, 99, 26, 29, 84, 25, 32, 17, 13, 59, 93, 32, 65, 39, 46, 11, 79],
    [70, 30, 52, 24, 33, 8, 51, 81, 66, 65, 63, 10, 55, 88, 76, 98, 88, 64, 76, 34, 52, 82, 52, 5, 36, 28, 90, 98, 26, 90],
    [5, 3, 31, 78, 31, 59, 99, 49, 50, 79, 28, 2, 100, 100, 56, 77, 16, 62, 21, 9, 38, 100, 59, 38, 2, 53, 9, 99, 47, 57],
    [40, 18, 52, 20, 14, 80, 51, 86, 11, 49, 66, 4, 16, 12, 85, 59, 20, 56, 94, 39, 8, 69, 83, 32, 68, 39, 37, 26, 88, 54],
    [16, 64, 14, 81, 35, 94, 8, 46, 87, 4, 53, 60, 10, 31, 87, 60, 41, 14, 49, 85, 17, 89, 75, 70, 77, 95, 3, 55, 58, 68],
])

scheduling_options = {
    'data': data
}

scheduling_manager = OPTModelManager('scheduling')
scheduling = scheduling_manager.get_example(1, **scheduling_options)

lb, ub = scheduling.get_bounds()
dim = data.shape[1]
NP = 20
x0 = initialize(NP, dim, ub, lb)
MaxIT = 200
fobj = lambda x: scheduling.evaluate(x)
option = opt_alg_options(x0, fobj, (lb, ub), NP, MaxIters=MaxIT)
optimizer = ExponentialTrigonometricOptAlg(option)
optimizer.run()
optimizer.print_optimal_result()
print(bm.argsort(optimizer.gbest))
scheduling.visualization(optimizer.gbest)