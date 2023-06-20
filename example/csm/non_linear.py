import numpy as np

from fealpy.opt import Problem
from fealpy.opt.PLBFGSAlg import PLBFGS



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






