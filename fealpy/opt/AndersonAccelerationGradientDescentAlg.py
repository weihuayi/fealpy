import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

"""
Reference
---------
https://en.wikipedia.org/wiki/Gradient_descent
"""


class AndersonAccelerationGradientDescentAlg:
    def __init__(self, problem, options=None):
        self.problem = problem
        self.options = options

        self.debug = True
        self.NF = 0  # 计算函数值和梯度的次数

        # compute the initial function value and gradient
        self.fun = problem['objective']
        self.x = problem['x0']
        self.f, self.g = self.fun(self.x)

    @classmethod
    def get_options(
            cls,
            MaxIters=500,
            MaxFunEvals=5000,
            NormGradTol=1e-6,
            FunValDiff=1e-6,
            StepLength=1,
            StepTol=1e-14,
            Disp=True,
            Output=False):

        options = {
                'MaxIters'          : MaxIters,
                'MaxFunEvals'       : MaxFunEvals,
                'NormGradTol'       : NormGradTol,
                'FunValDiff'        : FunValDiff,
                'StepLength'        : StepLength,
                'StepTol'           : StepTol,
                'Disp'              : Disp,
                'Output'            : Output
                }

        return options

    def run(self, queue=None, maxit=None):
        options = self.options
        alpha = options['StepLength']

        gnorm = norm(self.g)
        self.diff = np.Inf

        if options['Disp']:
            print("The initial F(x): %12.11g, grad:%12.11g, diff:%12.11g"%(self.f, gnorm, self.diff))

        if options['Output']:
            self.fun.output('', queue=queue)

        self.NF += 1

        if maxit is None:
            maxit = options['MaxFunEvals']

        for i in range(maxit):
            self.x -= alpha*self.g
            f, g = self.fun(self.x)
            self.diff = np.abs(f - self.f)
            self.f = f
            self.g = g

            gnorm = norm(self.g)

            if options['Disp']:
                print("Step %d with F(x): %12.11g, grad:%12.11g, diff:%12.11g"%(i, self.f, gnorm, self.diff))

            if options['Output']:
                self.fun.output(str(self.NF).zfill(6), queue=queue)

            self.NF += 1
            maxg = np.max(np.abs(self.g.flat))
            if (maxg < options['NormGradTol']):
                print("""
                The max norm of gradeint value : %12.11g (the tol  is %12.11g)
                The difference of function : %12.11g (the tol is %12.11g)
                """ % (
                    maxg, options['NormGradTol'],
                    self.diff, options['FunValDiff'])
                )
                break

        if options['Output']:
            self.fun.output('', queue=queue, stop=True)

        return self.x, self.f, self.g, self.diff

    def line_search(self):
        pass

    def show_linear_search(self, tag, x0,  d, fun, a, b):
        t = np.linspace(a, b, 40)
        N = t.shape[0]
        f = np.zeros(N)
        for i in range(N):
            f[i], g = fun(x0 + t[i]*d)

        fig = plt.figure()
        axes = fig.gca()
        axes.plot(t, f)
        plt.savefig(str(tag) + '.png')
        plt.close()

