import numpy as np

from fealpy.opt import Problem
from fealpy.opt.newton_raphson_optimizer import NewtonRaphsonOptimizer

class TwoNonlinearSpingsProblem(Problem):
    def __init__(self):
        x0 = np.zeros(2, dtype=np.float64)
        MaxIters = 20
        super().__init__(x0, self.energy, Preconditioner=self.tagnent_stiffness_matrix, MaxIters=MaxIters, )

    def energy(self, x):
        F = np.array([0, 100])
        U = np.array([25*x[0]**2 + 500/3*x[0]**3 + 50*(x[1]-x[0])**2 + 200/3*(x[1]-x[0])**3 - F[1]*x[1]])
        P = np.array([300*x[0]**2+400*x[0]*x[1]-200*x[1]**2+150*x[0]-100*x[1],
                      200*x[0]**2-400*x[0]*x[1]+200*x[1]**2-100*x[0]+100*x[1]-F[1]])
        return U, P

    def tagnent_stiffness_matrix(self, x):
        K = np.array([[600*x[0]+400*x[1]+150, 400*(x[0]-x[1])-100],
                      [400*(x[0]-x[1])-100, 400*x[1]-400*x[0]+100]])
        return np.linalg.inv(K) 

def two_nonlinear_springs_opt():
    problem = TwoNonlinearSpingsProblem()
    opt = NewtonRaphsonOptimizer(problem)
    x = opt.run()
    print("x:", x)
    print("energy:", problem.energy)



if __name__ == "__main__":
    two_nonlinear_springs_opt()






