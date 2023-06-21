import numpy as np

from .optimizer_base import Optimizer, Problem

class NewtonRaphsonOptimizer(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.P = problem.Preconditioner

    def run(self):
        problem = self.problem
        x = problem.x0
        f, grad = problem.objective(x)
        R = grad-np.array([0, 100])
        print("f:", f)
        print("grad:", grad)
        print("R:", R)
        for i in range(problem.MaxIters):
            print("x:", x)
            print("P:", self.P(x))
            du = -self.P(x) @ R
            print("du:", du)
            x += du
            _, grad = problem.objective(x)
            R = grad-np.array([0, 100])
            print("x:", x)
            f_new, grad_new = problem.objective(x)

            if np.abs(f_new - f) < problem.FunValDiff:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                break

            if np.linalg.norm(grad_new) < problem.NormGradTol:
                print(f"Convergence achieved after {i} iterations, the norm of gradient is less than NormGradTol")
                break

        f, grad = f_new, grad_new

        return x, f, grad 
