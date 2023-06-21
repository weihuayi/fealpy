import numpy as np

from .optimizer_base import Optimizer, Problem

class NewtonRaphsonOptimizer(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.P = problem.Preconditioner

    def run(self):
        problem = self.problem
        x0 = problem.x0
        f0, grad0 = problem.objective(x0)
        for i in range(problem.MaxIters):
            du = -self.P(x0) @ grad0
            x0 += du
            f, grad = problem.objective(x0)

            if np.abs(f - f0) < problem.FunValDiff:
                print()
                break

            if np.linalg.norm(grad) < problem.NormGradTol:
                print()
                break
        return x0 
