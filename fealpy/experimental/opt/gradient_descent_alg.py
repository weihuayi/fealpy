from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer, opt_alg_options

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
        obj = options['objective']
        x0 = options['x0']

        self.f, self.g = obj(x0)



        alpha = options['StepLength']

        gnorm = norm(self.g)
        self.diff = bm.inf 

        if options['Disp']:
            print("The initial F(x): %12.11g, grad:%12.11g, diff:%12.11g"%(self.f, gnorm, self.diff))

        if options['Output']:
            self.fun.output('', queue=queue)

        self.NF += 1

        if maxit is None:
            maxit = options['MaxFunEvals']

        for i in range(maxit):
            self.x -= alpha*self.g
            f, g = self.fun(self.x)
            self.diff = np.abs(f - self.f)
            self.f = f
            self.g = g

            gnorm = norm(self.g)

            self.NF += 1
            maxg = np.max(np.abs(self.g.flat))
            if (maxg < options['NormGradTol']):
                print("""
                The max norm of gradeint value : %12.11g (the tol  is %12.11g)
                The difference of function : %12.11g (the tol is %12.11g)
                """ % (
                    maxg, options['NormGradTol'],
                    self.diff, options['FunValDiff'])
                )
                break
        return self.x, self.f, self.g, self.diff
