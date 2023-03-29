import numpy as np
from numpy.linalg import norm, inv
from numpy.typing import NDArray
from scipy.linalg import cholesky

from .optimizer_base import Optimizer, Problem, Options
from .line_search import wolfe_line_search


class PLBFGS(Optimizer):
    def __init__(self, problem: Problem, options: Options) -> None:
        super().__init__(problem, options)

        self.pflag = True
        x = problem["x0"]
        ND = x.shape[0]
        self.S = np.zeros((ND, 0), dtype=np.float64)
        self.Y = np.zeros((ND, 0), dtype=np.float64)
        P = options['Preconditioner']
        self.L = cholesky((P+P.T)/2)

    @classmethod
    def get_options(
        cls, *,
        Preconditioner: NDArray,
        Display = False,
        MaxIterations = 500,
        StepTolerance = 1e-8,
        NormGradTolerance = 1e-6,
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
        N = self.S.shape[1]
        q = g
        rho = np.zeros((N, ), dtype=np.float64)
        alpha = np.zeros((N, ), dtype=np.float64)
        for i in range(N-1, -1, -1):
            s = self.S[:, i]
            y = self.Y[:, i]
            rho[i] = 1/np.sum(s*y)
            alpha[i] = np.sum(s*q)*rho[i]
            q = q - alpha[i]*y

        invL = inv(self.L)
        if self.pflag:
            r = invL.T@(invL@q)
        else:
            r = invL@q

        for i in range(0, N):
            s = self.S[:, i]
            y = self.Y[:, i]
            beta = rho[i] * (np.sum(y*r))
            r = r + (alpha[i] - beta)*s

        return r


    def run(self):
        options = self.options
        x = self.problem['x0']
        flag = True

        f, g = self.fun(x)
        gnorm = norm(g)
        pg = g

        alpha = 1

        if options['Display'] == 'iter':
            print("The initial status F(x): %12.11g, grad:%12.11g, diff:%12.11g"%(f, gnorm))


        flag = 0 # The convergence flag
        j = 0
        for i in range(1, options['MaxIterations']):
            d = -self.hessian_gradient_prod(g)
            gtd = np.sum(g*d)

            if np.abs(gtd) > 1e4:
                print('The norm of the desent direction is too big! Normalize it!\n')
                d = d/norm(d)
                gtd = np.sum(g*d)

            if gtd >= 0 or np.isnan(gtd):
                print('Not descent direction, quit at iteration %d, f = %g, gnorm = %5.1g\n'%(i, f, gnorm))
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
                    self.S = np.zeros((ND, 0), dtype=np.float64)
                    self.Y = np.zeros((ND, 0), dtype=np.float64)
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
            sty = np.sum(s*y)

            if sty < 0:
                print(f'bfgs: sty <= 0, skipping BFGS update at iteration {i}.')
            else:
                if i < options['NumGrad']:
                    self.S = np.hstack((self.S, s[:, None]))
                    self.Y = np.hstack((self.Y, y[:, None]))
                    j += 1
                else:
                    self.S[:, :-1] = self.S[:, 1:]
                    self.S[:, -1] = s
                    self.Y[:, :-1] = self.Y[:, 1:]
                    self.Y[:, -1] = y

        if flag == 0:
            flag = 3
        return x, f, g, flag
