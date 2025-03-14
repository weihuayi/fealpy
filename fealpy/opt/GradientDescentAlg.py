from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .optimizer_base import Optimizer, opt_alg_options

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

        if options['Print']:
            print(f'initial:  f = {f0}, gnorm = {gnorm}')

        for i in range(1,options["MaxIters"]):
            x1 =x0 - alpha*g0
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
                break

            if (gnorm < options['NormGradTol']):
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {self.problem.NormGradTol}")
                break

        return x0, f0, g0, diff
