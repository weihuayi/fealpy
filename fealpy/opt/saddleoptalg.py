import numpy as np
from numpy.linalg import norm

class SteepestDescentAlg:
    def __init__(self, problem, options=None):
        self.problem = problem
        self.options = options

        self.debug = True
        self.NF = 0 # 计算函数值和梯度的次数
        self.fun = problem['objective']
        self.x = problem['x0']
        self.f, self.g = self.fun(self.x) # 初始目标函数值和梯度值

    def run(self, maxit=None, q=None):
        problem = self.problem
        
        options = self.options
        alpha = options['StepLength']

        gnorm = norm(self.g)
        print("Initial energe: %12.11g, gnorm :%12.11g"%(self.f, gnorm))

        if maxit is None:
            maxit = options['MaxIters']

        for i in range(maxit):
            self.x[:, 0] += alpha*self.g[:, 0]
            self.x[:, 1] -= alpha*self.g[:, 1]
            f, g = self.fun(self.x)
            diff = np.abs(f - self.f)
            self.f = f
            self.g = g
            gnorm = norm(self.g)
            if options['Display'] is 'plot':
                self.fun.show()
                print("Step %d with energe: %12.11g, gnorm :%12.11g, energe diff:%12.11g"%(i, self.f, gnorm, diff))
            if diff < options['FunValDiff']:
                print('energe diff %12.11g is smaller than tol %g'%(diff,
                    options['FunValDiff']))
                break
        return self.x, self.f, self.g

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
        Display='plot'):

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
            'Display'           :Display # 'plot' or 'iter'
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

    def run(self, queue=None):
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

            if options['Display'] is 'plot':
                self.fun.show(queue=queue)
                print("Step %d with energe: %12.11g, gnorm :%12.11g"%(i, self.f, gnorm))
            self.f = f
            self.g = g

            gnorm = norm(self.g)
            if gnorm < options['NormGradTol']:
                print('gnorm %12.11g is smaller than tol %g'%(gnorm, options['NormGradTol']))
                break

        if options['Display'] is 'plot':
            self.fun.show(queue=queue, stop=True)
        return self.x, self.f, self.g
