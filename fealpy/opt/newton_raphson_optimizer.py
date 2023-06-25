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
        print("f:", f)
        print("grad:", grad)
        # 外力 [0, 100] 如何手动输入？
        R = np.array([0, 100]) - grad
        print("R:", R)
        print("----------------------------")
        for i in range(problem.MaxIters):
            print("Kt^-1:\n", self.P(x))
            du = self.P(x) @ R
            print("du:", du)
            x += du
            print("x:", x)
            _, grad = problem.objective(x)
            R = np.array([0, 100]) - grad
            print("R:", R)
#            print("x:", x)
            f_new, grad_new = problem.objective(x)
            print("f:", f_new)
            print("-------------------------------")

            if np.abs(f_new - f) < problem.FunValDiff:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                break

            if np.linalg.norm(grad_new) < problem.NormGradTol:
                print(f"Convergence achieved after {i} iterations, the norm of gradient is less than NormGradTol")
                break

        f, grad = f_new, grad_new

        return x, f, grad 
