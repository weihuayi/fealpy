from collections import deque
from typing import Union

import numpy as np
from numpy.linalg import norm, inv

from .optimizer_base import Optimizer, Problem
from .line_search import wolfe_line_search


class MatrixVectorProductGradientOptimizer(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.P = problem.Preconditioner
        assert self.P is not None

    def run(self):
        problem = self.problem
        x = problem.x0

        f, g = self.fun(x)
        gnorm = norm(g) 
        alpha = problem.StepLength  # 初始步长参数
        for i in range(1, problem.MaxIters):
            d = self.P@x
            gtd = np.dot(g, d)
            alpha, xalpha, falpha, galpha = wolfe_line_search(x, f, gtd, d, self.fun, alpha)

            if abs(f-falpha)<problem.FunValDiff:
            	x = xalpha
            	f = falpha
            	g = galpha
            	gnorm = norm(g)
            	break
            	
            x = xalpha
            f = falpha
            g = galpha
            gnorm = norm(g)

            if alpha < problem.StepLengthTol:
                break

            if problem.Print:
                print(f'current step {i}, StepLength = {alpha}, ', end='')
                print(f'nfval = {self.NF}, f = {f}, gnorm = {gnorm}')

            if gnorm < problem.NormGradTol:
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {problem.NormGradTol}")
                break

        return x, f, g

