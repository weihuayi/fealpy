class LineSearch:
    def search(self, x, objective, direction):
        raise NotImplementedError("Subclasses should implement this method.")


class ArmijoLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma=0.01):
        self.beta = beta
        self.sigma = sigma
    
    def search(self, x, objective, direction):
        alpha = 1
        f, g = objective(x)  
        while True:
            f_new, _ = objective(x + alpha * direction)  
            if f_new <= f + self.sigma * alpha * (g @ direction):
                break
            alpha *= self.beta
        return {'alpha': alpha}

class   PowellLineSearch(LineSearch):
    def __init__(self, beta=0.6, sigma1=0.01,sigma2=0.01, c=0.9):
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.c = c

    def search(self, x, objective, direction):
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
        return {'alpha': alpha}

class   GoldsteinLineSearch(LineSearch):  
   
    def __init__(self, beta=0.6, sigma=0.1):
        self.beta = beta
        self.sigma = sigma

    def search(self, x, objective, direction):
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
        return {'alpha': alpha}