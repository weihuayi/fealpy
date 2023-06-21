import numpy as np

from fealpy.opt import Problem
from fealpy.opt.PLBFGSAlg import PLBFGS

class TwoNonlinearSpingsProblem(Problem):
    def __init__(self):
        x0 = np.zeros(2, dtype=np.float64)
        super().__init__(x0, self.energy, Preconditioner=self.tagnent_stiffness_matrix)

    def energy(self, x):
        f = np.array([0, 100])
        U = np.array([25*x[0]**2 + 500/3*x[0]**3 + 50*(x[1]-x[0])**2 + 200/3*(x[1]-x[0])**3 - f[1]*x[1]])
        P = np.array([300*x[0]**2+400*x[0]*x[1]-200*x[1]**2+150*x[0]-100*x[1],
                     200*x[0]**2-400*x[0]*x[1]+200*x[1]**2-100*x[0]+100*x[1]])
        return U, P

    def tagnent_stiffness_matrix(self, x):
        K = np.array([[600*x[0]+400*x[1]+150, 400*(x[0]-x[1])-100],
                       [400*(x[0]-x[1])-100, 400*x[1]-400*x[0]+100]])
        return np.inv(K) 

def two_nonlinear_springs_opt():
    x0 = np.zeros(2, dtype=np.float64)
    F = np.array([0.0, 100.0])
    def energy(x):
        U = 25.0 * x[0]**2 + 500 * x[0]**3 / 3.0 + \
                50.0 * (x[1] - x[0])**2 + 200.0 * (x[1] - x[0])**3 / 2.0 - F[1]*x[1]

        P = np.array([
            300*x[0]**2+400*x[0]*x[1]-200*x[1]**2+150*x[0]-100*x[1],
            200*x[0]**2-400*x[0]*x[1]+200*x[1]**2-100*x[0]+100*x[1]-F[1]
            ])
        return U, P

    problem = Problem(x0=x0, objective=energy)
    opt = PLBFGS(problem)
    x, _, _, _ = opt.run()
    print("x:", x)



if __name__ == "__main__":
    two_nonlinear_springs_opt()






