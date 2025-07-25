from fealpy.opt import NondominatedSortingGeneticAlgIII
from fealpy.opt.benchmark.multi_benchmark import get_dtlz
from fealpy.opt.opt_function import generate_reference_points_double_layer, initialize

M = 3
dim = 7
func = get_dtlz('DTLZ1', M=M, V=dim)
fobj = func['fobj']
lb, ub = func['domain']
H1 = 3
H2 = 2

zr = generate_reference_points_double_layer(M=M, H1=H1, H2=H2)
x = initialize(zr.shape[0], dim, ub, lb)
test = NondominatedSortingGeneticAlgIII(
    M=M, 
    dim=dim, 
    lb=lb, 
    ub=ub, 
    zr=zr, 
    x=x, 
    fobj=fobj
)
pop = test.run()
pass