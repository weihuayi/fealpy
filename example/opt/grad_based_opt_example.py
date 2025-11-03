import argparse
from fealpy.backend import backend_manager as bm
from fealpy.opt import GradientDescent,PLBFGS, PNLCG

parser = argparse.ArgumentParser(description=
        '''
        基于梯度的优化算法示例
        ''')
parser.add_argument('--exam',
                    default='e1',type=str,
                    help='''
                    e1: 二次函数优化
                    e2: Rosenbrock 函数优化
                        ''')

parser.add_argument('--backend',
                    default='numpy',type=str,
                    help='''
                    选择计算后端，支持 'numpy', 'pytorch', 'jax'
                    ''')
args = parser.parse_args()
bm.set_backend(args.backend)

def quadratic_function(x):
    f = (x[0]-3)**2+10*(x[1]-2)**2
    gradf = bm.array([2*(x[0]-3),20*(x[1]-2)])
    return f,gradf

def rosenbrock_function(x):
    f = (1 - x[0])**2 + 100*(x[1] - x[0]*x[0])**2
    dx = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    dy = 200*(x[1] - x[0]**2)
    gradf = bm.array([dx, dy])
    return f,gradf

if args.exam == 'e1':
    func = quadratic_function
    x0 = bm.array([0.0, 0.0])
if args.exam == 'e2':
    func = rosenbrock_function
    x0 = bm.array([-1.2, 1.0])

options = GradientDescent.get_options(
        x0 = x0,
        objective=func,
        MaxIters = 500,
        FunValDiff = 1e-6,
        StepLength=1,
        Print = False)

optGD = GradientDescent(options)
optLBFGS = PLBFGS(options)
optNLCG = PNLCG(options)

print("GradientDescent:")
xopt_GD,fopt_GD,gradopt_GD = optGD.run()
print("gradient descent optimal x:", xopt_GD)
print("gradient descent optimal f:", fopt_GD)

print("\nLBFGS:")
xopt_LBFGS,fopt_LBFGS,gradopt_LBFGS = optLBFGS.run()
print("LBFGS optimal x:", xopt_LBFGS)
print("LBFGS optimal f:", fopt_LBFGS)

print("\nNLCG:")
xopt_NLCG,fopt_NLCG,gradopt_NLCG = optNLCG.run()
print("NLCG optimal x:", xopt_NLCG)
print("NLCG optimal f:", fopt_NLCG)

