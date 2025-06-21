import numpy as np
from .optimizer_base import Optimizer, Problem
from .line_search import get_linesearch


class ModifiedNewtonRaphsonOptimizer(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.P = problem.Preconditioner

    def run(self):
        problem = self.problem
        x = problem.x0
        f, gradf = problem.objective(x)
        alpha = problem.StepLength
        for i in range(problem.MaxIters):
            du = -self.P(x) @ gradf
            if problem.Linesearch is None:
                x += du
                f_new, gradf_new = problem.objective(x)
            else:
                gtd = np.dot(gradf,du)
                func = get_linesearch(problem.Linesearch)
                alpha,x,f_new,gradf_new = func(x0=x,f=f,s=gtd,d=du,fun=problem.objective,alpha0=alpha)

            if np.abs(f_new - f) < problem.FunValDiff:
                print(f"Convergence achieved after {i+1} iterations, the function value difference is less than FunValDiff")
                break

            if np.linalg.norm(gradf_new) < problem.NormGradTol:
                print(f"Convergence achieved after {i+1} iterations, the norm of gradient is less than NormGradTol")
                break

            alpha,x, f, gradf = alpha, x, f_new, gradf_new

        return x, f, gradf 


