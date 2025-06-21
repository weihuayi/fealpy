import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize


class AndersonAccelerationAlg:
    def __init__(self, problem, options=None):
        self.problem = problem
        self.options = options

        self.debug = True
        self.NF = 0  # 计算函数值和梯度的次数
        self.fun = problem['objective']
        self.x = problem['x0']
        self.f, self.g = self.fun(self.x)  # 初始目标函数值和梯度值
        g = self.g.copy()
        g[:, 1] *= -1
        self.F = [g]
        self.m = 5

        # compute one step
        self.x[:, 0] += self.g[:, 0]
        self.x[:, 1] -= self.g[:, 1]
        self.f, self.g = self.fun(self.x)
        g = self.g.copy()
        g[:, 1] *= -1
        self.F += [g]

    def run(self, queue=None, maxit=None):
        options = self.options
        alpha = options['StepLength']

        gnorm = norm(self.g)
        N = len(self.g)
        self.diff = np.Inf
        if options['Output']:
            print("Step %d with energe: %12.11g, gnorm :%12.11g, energe diff:%12.11g"%(self.NF, self.f, gnorm, self.diff))
            self.fun.output('', queue=queue)

        self.NF += 1

        if maxit is None:
            maxit = options['MaxFunEvals']

        for i in range(maxit):
            alpha = self.minfun()
            self.x += sum(map(lambda x: x[0]*x[1], zip(alpha, self.F)))
            f, g = self.fun(self.x)
            self.diff = np.abs(f - self.f)
            self.f = f
            self.g = g
            g = g.copy()
            g[:, 1] *= -1
            if len(self.F) < self.m:
                self.F += [g]
            else:
                self.F = self.F[1:] + [g]
            gnorm = norm(self.g)
            maxg = np.sqrt(np.max(np.sum(self.g**2, axis=-1)))
            if options['Output']:
                print("Step %d with energe: %12.11g, maxgnorm :%12.11g, energe diff:%12.11g"%(self.NF, self.f, maxg, self.diff))
                self.fun.output(str(self.NF).zfill(6), queue=queue)
            self.NF += 1

            if (maxg < options['NormGradTol']) or (self.diff < options['FunValDiff']):
                print("""
                The max norm of gradeint value : %12.11g (the tol  is %12.11g)
                The difference of function : %12.11g (the tol is %12.11g)
                """ % (
                    maxg, options['NormGradTol'],
                    self.diff, options['FunValDiff'])
                )
                break

        self.fun.output('', queue=queue, stop=True)

        return self.x, self.f, self.g, self.diff

    def minfun(self):
        n = len(self.F)
        x0 = 1/n*np.ones(n, dtype=np.float)
        eq_cons = {'type': 'eq',
                   'fun': lambda x: np.array([sum(x) - 1]),
                   'jac': lambda x: np.ones(n)}
        res = minimize(
                lambda x: norm(sum(map(lambda y: y[0]*y[1], zip(x, self.F)))),
                x0, method='SLSQP', constraints=[eq_cons],
                options={'ftol': 1e-9, 'disp': True})
        return res.x

class SteepestDescentAlg:
    def __init__(self, problem, options=None):
        self.problem = problem
        self.options = options

        self.debug = True
        self.NF = 0  # 计算函数值和梯度的次数
        self.fun = problem['objective']
        self.x = problem['x0']
        self.f, self.g = self.fun(self.x)  # 初始目标函数值和梯度值

    def run(self, queue=None, maxit=None, eta_ref = None):
        options = self.options
        alpha = options['StepLength']

        gnorm = norm(self.g)
        self.diff = np.Inf

        if options['Output']:
            print("Step %d with energe: %12.11g, gnorm :%12.11g, energe diff:%12.11g"%(self.NF, self.f, gnorm, self.diff))
            self.fun.output('', queue=queue)

        self.NF += 1

        if maxit is None:
            maxit = options['MaxFunEvals']

        for i in range(maxit):
            self.x[:, 0] += alpha*self.g[:, 0]
            self.x[:, 1] -= alpha*self.g[:, 1]
            f, g = self.fun(self.x)
            self.diff = np.abs(f - self.f)

            self.f = f
            self.g = g
            gnorm = norm(self.g)

            maxg = np.sqrt(np.max(np.sum(self.g**2, axis=-1)))
            if options['Output']:
                print("Step %d with energe: %12.11g, maxgnorm :%12.11g, energe diff:%12.11g"%(self.NF, self.f, maxg, self.diff))
                self.fun.output(str(self.NF).zfill(6), queue=queue)
            self.NF += 1

            if (maxg < options['NormGradTol']) or (self.diff < options['FunValDiff']):
                print("""
                The max norm of gradeint value : %12.11g (the tol  is %12.11g)
                The difference of function : %12.11g (the tol is %12.11g)
                """ % (
                    maxg, options['NormGradTol'],
                    self.diff, options['FunValDiff'])
                )
                break

            #TODO
            if eta_ref is not None:
                eta_ref = self.fun.eta_ref
                if eta_ref < options['etarefTol']:
                    break

        self.fun.output('', queue=queue, stop=True)

        return self.x, self.f, self.g, self.diff

    def step(self, alpha=2):
        self.x[:, 0] += alpha*self.g[:, 0]
        self.x[:, 1] -= alpha*self.g[:, 1]
        self.f, self.g = self.fun(self.x)

def HCG_options(
        MaxIters=500,
        MaxFunEvals=5000,
        NormGradTol=1e-6,
        FunValDiff=1e-6,
        StepLength=1,
        HybridFactor=0.618,
        StepTol=1e-14,
        RestartSteps=0, 
        Adaptive=False,
        ScaleFactors=(0.618, 0.618, 1/0.618),
        Output= True):

    options = {
            'MaxIters'          :MaxIters,
            'MaxFunEvals'       :MaxFunEvals,
            'NormGradTol'       :NormGradTol,
            'FunValDiff'        :FunValDiff,
            'StepLength'        :StepLength,
            'HybridFactor'      :HybridFactor,
            'StepTol'           :StepTol,
            'RestartSteps'      :RestartSteps,
            'Adaptive'          :Adaptive,
            'ScaleFactors'      :ScaleFactors,
            'Output'            :Output 
            }
    return options

class HybridConjugateGradientAlg:
    def __init__(self, problem, options=None):
        self.problem = problem
        self.debug = True
        self.NF = 0 # 计算函数值和梯度的次数

        self.fun = problem['objective']
        self.x = problem['x0']
        self.f, self.g = self.fun(self.x) # 初始目标函数值和梯度值
        self.NF += 1

        if options == None:
            self.options = HCG_options()
        else:
            self.options = options

    def run(self, queue=None, maxit=None, ):
        problem = self.problem
        options = self.options 
        alpha = options['StepLength']
        beta = options['HybridFactor'] 
        r = options['RestartSteps'] # 0 <= r <= 1
        s0 = options['ScaleFactors'][0] # 0 < s0 < 1
        s1 = options['ScaleFactors'][1] # 0 < s1 < 1
        s2 = options['ScaleFactors'][2] # s2 > 1
        adaptive = options['Adaptive']

        gnorm = norm(self.g)
        print("The initial energe is %12.11g, the norm of grad is %12.11g"%(self.f, gnorm))

        if maxit is None:
            maxit = options['MaxIters']

        # the initial direction
        d0 =  self.g[:, 0]
        d1 = -self.g[:, 1]
        for i in range(1, options['MaxIters']+1):
            self.x[:, 0] += alpha*d0
            self.x[:, 1] += alpha*d1
            f, g = self.fun(self.x)

            if adaptive: # adaptive adjust hybrid factor 
                if f < self.f:
                    beta = min(s2*beta, 1)
                else:
                    beta = max(s1*beta, s0)

            # update direction
            gamma0 = np.sum(g[:, 0]*g[:, 0])/np.sum(self.g[:, 0]*self.g[:, 0])
            gamma1 = np.sum(g[:, 1]*g[:, 1])/np.sum(self.g[:, 1]*self.g[:, 1])
            if r == 0 or i%r != 0: # no restart 
                d0 =  (1-beta)*g[:, 0] + beta*gamma0*d0
                d1 = -(1-beta)*g[:, 1] + beta*gamma1*d1
            else: # restart 
                d0 =  g[:, 0]
                d1 = -g[:, 1]

            diff = np.abs(f - self.f)
            self.f = f
            self.g = g

            gnorm = norm(self.g)
            if options['Output']:
                print("Step %d with energe: %12.11g, gnorm :%12.11g, energe diff:%12.11g"%(i, self.f, gnorm, diff))
                self.fun.output(str(i), queue=queue)

            if (gnorm < options['NormGradTol']) or (diff < options['FunValDiff']):
                print("""
                The norm of gradeint value : %12.11g (the tol  is %12.11g)
                The difference of function : %12.11g (the tol is %12.11g)
                """%(gnorm, options['NormGradTol'], diff,
                    options['FunValDiff']))
                break


        self.fun.output('', queue=queue, stop=True)

        return self.x, self.f, self.g


