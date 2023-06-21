import numpy as np

from .optimizer_base import Optimizer, Problem

class NewtonRaphsonOptimizer(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.P = problem.Preconditioner
        assert self.P is not None

    def run(self):
        problem = self.problem
        x = problem.x0
        iter = 0
        xold = x
        P = problem.energy(x)[1]
        R = problem.F - P
        conv = np.sum(R**2)/(1+np.sum(problem.F**2))
        c = 0

        while conv > problem.StepLengthTol and iter < problem.MaxIters:
            Kt = calculate_Kt(u)
            delu = np.linalg.solve(Kt, R)
            u = uold + delu
            P = calculate_P(u)
            R = f - P
            conv = np.sum(R**2)/(1+np.sum(f**2))
            c = np.abs(u_exact[1] - u[1])/np.abs(u_exact[1] - uold[1])**2 if iter > 0 else 0
            uold = u
            iter += 1
        return u
