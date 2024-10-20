from fealpy.backend import backend_manager as bm
from fealpy.opt.optimizer_base import Optimizer, opt_alg_options
from fealpy.opt.line_search_rules import *
"""
Reference
---------
https://en.wikipedia.org/wiki/Gradient_descent
"""


class GradientDescentAlg(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)
    

    def run(self, queue=None, maxit=None):
        options = self.options
        if self.options['LineSearch'] is None:
            self.options['LineSearch'] = PowellLineSearch()  # 默认使用 Powell 线搜索
        self.line_search_method = options['LineSearch']
        self.x = [] # 保存迭代点
        x0 = options['x0']
        
        self.x.append(x0) 
        self.f, self.g = self.fun(x0)

        alpha = options['StepLength']

        gnorm = bm.linalg.norm(self.g)
        self.diff = bm.inf 

        if maxit is None:
           maxit = options['MaxFunEvals']
        C = self.f
        for i in range(maxit):
            
            if isinstance(self.line_search_method, ZhangHagerLineSearch):
                alpha,C = self.line_search_method.search(self.x, self.fun, -self.g, C)
            else:
                alpha = self.line_search_method.search(self.x, self.fun, -self.g)
            x_new = self.x[-1]- alpha*self.g
            self.x.append(x_new)
            f, g = self.fun(self.x[-1])
            self.diff = bm.abs(f - self.f)
            self.f = f
            self.g = g
            
            gnorm = bm.linalg.norm(self.g)
            maxg = bm.max(bm.abs(self.g.flatten()))
            
            if (maxg < options['NormGradTol']):
                print("""
                The max norm of gradeint value : %12.11g (the tol  is %12.11g)
                The difference of function : %12.11g (the tol is %12.11g)
                """ % (
                    maxg, options['NormGradTol'],
                    self.diff, options['FunValDiff'])
                )
                break
        return self.x[-1], self.f, self.g, self.diff
