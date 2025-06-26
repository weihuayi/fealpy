from fealpy.backend import backend_manager as bm
from fealpy.opt import PLBFGS,PNLCG
from fealpy.opt import GradientDescent as GD


backend = ['numpy','pytorch','jax']

for be in backend:
    print(f'Backend: {be}\n')
    bm.set_backend(be)

    def q(x):
        f = x[0]**4+x[1]**4-x[0]**2-x[1]**2-2*x[0]*x[1]
        g = bm.array([4*x[0]**3-2*x[0]-2*x[1],4*x[1]**3-2*x[1]-2*x[0]],dtype =
                     bm.float64)
        return f, g

    x0 = bm.array([2.0, 3.0],dtype=bm.float64)
#梯度下降法
    options = GD.get_options(x0=x0,objective=q)
    options['StepLength'] = 0.01
    options['Print'] = False
    gd = GD(options)
    x,f,g,_ = gd.run()
    print(f'极小值点:{x},极小值:{f},梯度:{g},函数与梯度计算次数:{gd.NF}\n')

#lbfgs算法
    options = PLBFGS.get_options(x0=x0,objective=q)
    options['Print'] = False
    lbfgs = PLBFGS(options)
    x,f,g,_ = lbfgs.run()
    print(f'极小值点:{x},极小值:{f},梯度:{g},函数与梯度计算次数:{lbfgs.NF}\n')

#nlcg算法
    options = PNLCG.get_options(x0=x0,objective=q)
    options['Print'] = False
    nlcg = PNLCG(options)
    x,f,g = nlcg.run()
    print(f'极小值点:{x},极小值:{f},梯度:{g},函数与梯度计算次数:{nlcg.NF}\n')

