from fealpy.backend import backend_manager as bm
from scipy.optimize import minimize_scalar

class LineSearch:
    def search(self, x, objective, direction):
        raise NotImplementedError("Subclasses should implement this method.")

class StrongWolfeLineSearch(LineSearch):
    def __init__(self):
        pass

    def zoom(self,x,s,d,objective,alpha_0,alpha_1,f0,fl,c1,c2):
        iter_ = 0
        while iter_ < 20:
            alpha = (alpha_0 + alpha_1)/2
            xc = x + alpha*d
            fc, gc = objective(xc)
            if (fc > f0 + c1*alpha*s)\
            or (fc >= fl):
                alpha_1 = alpha
            else:
                sc = bm.dot(gc,d)
                if bm.abs(sc) <= -c2*s:
                    return alpha, xc, fc, gc

                if sc*(alpha_1 - alpha_0) >= 0:
                    alpha_1 = alpha_0
                    fl = fc
                alpha_0 = alpha

            iter_ += 1
        return alpha, xc, fc, gc

    def search(self,x0,f,s,d,objective,alpha0):
        c1,c2 = 0.001,0.1
        alpha = alpha0
        alpha_0 = 0.0
        alpha_1 = alpha

        fx = f
        f0 = f
        iter_ = 0

        while iter_ < 10:
            xc = x0 + alpha_1*d
            fc, gc = objective(xc)
            sc = bm.dot(gc,d)

            if (fc > f0 + c1*alpha_1*s)\
            or (
                (iter_ > 0) and (fc >= fx)
            ):
                alpha, xc, fc, gc = self.zoom(
                    x0, s, d, objective, alpha_0, alpha_1, f0, fc, c1, c2
                )
                break

            if bm.abs(sc) <= -c2*s:
                alpha = alpha_1
                break

            if (sc >= 0):
                alpha, xc, fc, gc = self.zoom(
                    x0, s, d, objective, alpha_1, alpha_0, f0, fc, c1, c2
                )
                break

            alpha_0 = alpha_1
            alpha_1 = min(10, 3*alpha_1)
            fx = fc
            iter_ = iter_ + 1
        return alpha, xc, fc, gc

class ArmijoLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma=0.001,max_iter=20):
        self.beta = beta
        self.sigma = sigma
        self.max_iter = max_iter
    
    def search(self, x, objective, direction,alpha):
        alpha = min(1.0, 2*alpha)
        f, g = objective(x)
        gtd = bm.dot(g, direction)
        for _iter in range(self.max_iter):
            f_new, _ = objective(x + alpha*direction)  
            if f_new <= f + self.sigma*alpha*gtd:
                return alpha
            alpha *= self.beta
        return alpha

class PowellLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma1=0.001,sigma2=0.01, c=0.9,max_iter=20):
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.c = c
        self.max_iter = max_iter

    def search(self, x, objective, direction,alpha):
        alpha = min(1,2*alpha)
        f, g = objective(x)
        gtd = bm.dot(g, direction)
        while True:
            f_new, g_new = objective(x + alpha*direction)  
            if f_new <= f + self.sigma1*alpha*gtd:
                break
            alpha *= self.beta
        k = alpha
        while True:
            f_new, g_new = objective(x + alpha * direction)
            if g_new @ direction >= self.c * (g @ direction):
                break
            alpha += self.sigma2  * (k/self.beta - k)
        return alpha

class GoldsteinLineSearch(LineSearch):   
    def __init__(self, beta=0.6, sigma=0.1):
        self.beta = beta
        self.sigma = sigma

    def search(self, x, objective, direction,alpha):
        alpha = min(1,2*alpha)
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

class ZhangHagerLineSearch(LineSearch): 
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
