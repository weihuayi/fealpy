import numpy as np

from .optimizer_base import Optimizer, Problem

class Broyden(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.P = problem.Preconditioner

    def run(self):
        problem = self.problem
        x = problem.x0
        f, gradf = problem.objective(x)
        K_inv = -self.P(x)
        for i in range(problem.MaxIters):
            du = -K_inv @ gradf
            x += du
            _, gradf = problem.objective(x)
            f_new, grad_f_new = problem.objective(x)

            if np.abs(f_new - f) < problem.FunValDiff:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                break

            if np.linalg.norm(grad_f_new) < problem.NormGradTol:
                print(f"Convergence achieved after {i} iterations, the norm of gradient is less than NormGradTol")
                break

        f, gradf = f_new, grad_f_new

        return x, f, gradf 
