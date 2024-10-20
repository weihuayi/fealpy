
from scipy.optimize import minimize_scalar

class LineSearch:
    def search(self, x, objective, direction):
        raise NotImplementedError("Subclasses should implement this method.")


class ArmijoLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma=0.01):
        self.beta = beta
        self.sigma = sigma
    
    def search(self, x, objective, direction):
        x = x[-1]
        alpha = 1
        f, g = objective(x)  
        while True:
            f_new, _ = objective(x + alpha * direction)  
            if f_new <= f + self.sigma * alpha * (g @ direction):
                break
            alpha *= self.beta
        return alpha

class   PowellLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma1=0.01,sigma2=0.01, c=0.9):
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.c = c

    def search(self, x, objective, direction):
        x = x[-1]
        alpha = 1
        f, g = objective(x)  
        while True:
            f_new, g_new = objective(x + alpha * direction)  
            if f_new <= f + self.sigma1 * alpha * (g @ direction):
                break
            alpha *= self.beta
        k = alpha
        while True:
            f_new, g_new = objective(x + alpha * direction)
            if g_new @ direction >= self.c * (g @ direction):
                break
            alpha += self.sigma2  * (k/self.beta - k)
        return alpha

class   GoldsteinLineSearch(LineSearch):  
   
    def __init__(self, beta=0.6, sigma=0.1):
        self.beta = beta
        self.sigma = sigma

    def search(self, x, objective, direction):
        x = x[-1]
        alpha = 1
        f, g = objective(x)  
        while True:
            f_new, _ = objective(x + alpha * direction) 
            if f_new <= f + self.sigma * alpha * (g @ direction):
                break
            alpha *= self.beta
        while True:
            f_new, _ = objective(x + alpha * direction)
            if f_new > f + (1-self.sigma) * alpha * (g @ direction):
                break
            alpha *= self.beta
        return alpha
    
class GrippoLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma=0.01, step=5):
        self.beta = beta
        self.sigma = sigma
        self.step = step

    def search(self, x, objective, direction):
        alpha = 1
        n = len(x)
        f, g = objective(x[-1])

    # 获取最近的步长
        if n < self.step:
            f_max = max(objective(x[i])[0] for i in range(n))
        else:
            f_max = max(objective(x[i])[0] for i in range(n-self.step, n))

        while True:
            f_new, _ = objective(x[-1] + alpha * direction)
            if f_new <= f_max + self.sigma * alpha * (g @ direction):
                break
            alpha *= self.beta

        return alpha

class   ZhangHagerLineSearch(LineSearch): 
    def __init__(self, gamma=0.85, eta=0.2, c=1e-3):
        self.gamma = gamma
        self.eta = eta
        self.c = c

    def search(self, x, objective, direction, C):
        alpha = 1
        iter = len(x)
        f, g = objective(x[-1])  

        while True:
            f_new, _ = objective(x[-1] + alpha * direction)  
            if f_new <= C + self.c * alpha * (g @ direction):
                break
            alpha *= self.eta
        Qp = (1 - pow(self.gamma, iter))/(1 - self.gamma)
        C = (self.gamma * Qp * C + f) / (self.gamma * Qp + 1)   
        return alpha, C

class ExactLineSearch(LineSearch):

    def __init__(self, step_length_bounds=(0, 0.3)):
        self.step_length_bounds = step_length_bounds  # 步长范围
       

    def search(self, x,objective, direction):

        def f(p):
            return objective(p)[0]
        def func(alpha):
            return f(x[-1] + alpha * direction)
        res = minimize_scalar(func,bounds=self.step_length_bounds, method='bounded')

        return res.x
