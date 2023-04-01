from collections import deque
from typing import Union

import numpy as np
from numpy.linalg import norm, inv
from numpy.typing import NDArray

from scipy.sparse import spmatrix  # 代表 scipy 的稀疏矩阵
from scipy.sparse.linalg import LinearOperator

from .optimizer_base import Optimizer, Problem, Options
from .line_search import wolfe_line_search

# 表示变量可以是 LinearOperator、稀疏矩阵或密集矩阵
MatrixLike = Union[LinearOperator, spmatrix, np.ndarray, None]

class PLBFGS(Optimizer):
    def __init__(self, problem: Problem, options: Options) -> None:
        super().__init__(problem, options)

        self.S = deque() 
        self.Y = deque() 
        # if self.P is None, 表示没有预条件子
        self.P: MatrixLike = options['Preconditioner']

    @classmethod
    def get_options(
        cls, *,
        Preconditioner: MatrixLike,
        Display: bool = False,
        MaxIterations: int = 500,
        StepTolerance: float = 1e-8,
        NormGradTolerance: float = 1e-6,
        NumGrad = 10,
    ) -> Options:

        options = {
            "Preconditioner": Preconditioner,
            "Display": Display,
            "MaxIterations": MaxIterations,
            "StepTolerance": StepTolerance,
            "NormGradTolerance": NormGradTolerance,
            "NumGrad": NumGrad
        }

        return super().get_options(**options)


    def hessian_gradient_prod(self, g: NDArray) -> NDArray:
        N = len(self.S)
        q = g
        rho = np.zeros((N, ), dtype=np.float64)
        alpha = np.zeros((N, ), dtype=np.float64)
        for i in range(N-1, -1, -1):
            rho[i] = 1/np.dot(self.S[i], self.Y[i])
            alpha[i] = np.dot(self.S[i], q)*rho[i]
            q = q - alpha[i]*self.Y[i]

        if self.P is not None:
            r = self.P@q
        else:
            r = q

        for i in range(0, N):
            beta = rho[i] * (np.dot(self.Y[i], r))
            r = r + (alpha[i] - beta)*self.S[i]

        return r


    def run(self):
        options = self.options
        x = self.problem['x0']
        flag = True

        f, g = self.fun(x)
        gnorm = norm(g)
        pg = g

        alpha = 1
        diff = np.inf

        if options['Display'] == 'iter':
            print(f"The initial status F(x): {f:%12.11g}, grad:{gnorm:%12.11g}, diff:{diff:%12.11g}")


        flag = 0 # The convergence flag
        j = 0
        for i in range(1, options['MaxIterations']):
            d = -self.hessian_gradient_prod(g)
            gtd = np.dot(g, d)

            if gtd >= 0 or np.isnan(gtd):
                print('Not descent direction, quit at iteration {i} witht statt {f:%12.11g}, grad:{gnorm:%12.11g}, diff:{diff:%12.11g}')
                break

            pg = g

            alpha, xalpha, falpha, galpha = wolfe_line_search(x, f, gtd, d, self.fun, alpha)
            if np.abs(falpha - f) < 1e-4:
                flag = 1
                break

            if alpha > self.options['StepTolerance']:
                x = xalpha
                f = falpha
                g = galpha
                gnorm = norm(g)
            else:
                if options['Display'] == 'iter':
                    print('The step length alpha %g is smaller than tolerance %g!\n'%(alpha, options['StepTolerance']))

                if j == 0:
                    flag = 2
                    break
                else:
                    alpha = 1
                    if options['Display'] == 'iter':
                        print(f'restart with step length {alpha}.')
                    ND = x.shape[0]
                    self.S = deque()
                    self.Y = deque() 
                    j = 0
                    continue


            if options['Display'] == 'iter':
                print(f'current step {i}, alpha = {alpha}, ', end='')
                print(f'nfval = {self.NF}, maxd = {np.max(np.abs(x))}, f = {f}, gnorm = {gnorm}')

            if gnorm < options['NormGradTolerance']:
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {options['NormGradTolerance']}")
                flag = 1 # convergence
                break

            s = alpha*d
            y = g - pg
            sty = np.dot(s, y)

            if sty < 0:
                print(f'bfgs: sty <= 0, skipping BFGS update at iteration {i}.')
            else:
                if i < options['NumGrad']:
                    self.S.append(s) 
                    self.Y.append(y)
                    j += 1
                else:
                    self.S.popleft()
                    self.S.append(s)
                    self.Y.popleft()
                    self.Y.append(y)

        if flag == 0:
            flag = 3
        return x, f, g, flag
