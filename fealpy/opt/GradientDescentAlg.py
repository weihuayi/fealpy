from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .optimizer_base import Optimizer, opt_alg_options
from .line_search_rules import ArmijoLineSearch

class GradientDescent(Optimizer):
    def __init__(self,options):
        super().__init__(options)

    @classmethod
    def get_options(
            cls,
            x0: TensorLike,
            objective,
            MaxIters=500,
            MaxFunEvals=5000,
            NormGradTol=1e-6,
            FunValDiff=1e-6,
            StepLength=1,
            StepLengthTol=1e-14,
            Print=True):

        return opt_alg_options(
            x0=x0,
            objective=objective,
            MaxIters=MaxIters,
            StepLength=StepLength,
            StepLengthTol=StepLengthTol,
            NormGradTol=NormGradTol,
            Print = Print
        )

    def run(self):
        options = self.options
        x0 = options['x0']
        alpha = options['StepLength']
        f0,g0 = self.fun(x0)
        gnorm = bm.linalg.norm(g0)
        Armijo = ArmijoLineSearch() 
        if options['Print']:
            print(f'initial:  f = {f0}, gnorm = {gnorm}')

        for i in range(1,options["MaxIters"]):
            direction = -g0
            alpha = Armijo.search(x0,self.fun, direction, alpha)
            x1 =x0 + alpha*direction
            f1, g1 = self.fun(x1)
            diff = bm.abs(f1 - f0)
            f0 = f1
            g0 = g1
            x0 = x1
            gnorm = bm.linalg.norm(g0)

            if options["Print"]:
                print(f'current step {i}, StepLength = {alpha}, ', end='')
                print(f'nfval = {self.NF}, f = {f0}, gnorm = {gnorm}')

            if diff < options["FunValDiff"]:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                return x0, f0, g0

            if (gnorm < options['NormGradTol']):
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {self.problem.NormGradTol}")
                return x0, f0, g0
        print(f"Reached the Maximum number of iterations {options['MaxIters']} times")
        return x0, f0, g0
