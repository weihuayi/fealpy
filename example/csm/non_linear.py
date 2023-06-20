import numpy as np

from fealpy.opt import Problem
from scipy.sparse.linalg import LinearOperator

from fealpy.opt.MatrixVectorProductGradientOptimizer import MatrixVectorProductGradientOptimizer

class TwoNonlinearSpingsProblem(Problem):
    def __init__(self):
        self.F = np.array([0.0, 100.0])
        x0 = np.zeros(2, dtype=np.float64)
        super().__init__(x0, self.energy)
        self.MaxIters = 1000 
        self.Preconditioner = self.tagnent_stiffness_matrix

    def energy(self, x):
        F = self.F
        U = 25.0 * x[0]**2 + 500 * x[0]**3 / 3.0 + \
                50.0 * (x[1] - x[0])**2 + 200.0 * (x[1] - x[0])**3 / 2.0 - F[1]*x[1]

        P = np.array([
            300*x[0]**2+400*x[0]*x[1]-200*x[1]**2+150*x[0]-100*x[1],
            200*x[0]**2-400*x[0]*x[1]+200*x[1]**2-100*x[0]+100*x[1]
            ])
        return U, P

    def tagnent_stiffness_matrix(self, x):
        Kt = np.array([
            [600*x[0]+400*x[1]+150, 400*(x[0]-x[1])-100],
            [400*(x[0]-x[1])-100, 400*x[1]-400*x[0]+100]
            ])
        return Kt

def two_nonlinear_springs_opt():
    problem = TwoNonlinearSpingsProblem()
    NDof = len(problem.x0)
    problem.Preconditioner = LinearOperator((NDof, NDof), problem.newton_raphson_preconditioner)
    problem.StepLength = 1.0
    opt = MatrixVectorProductGradientOptimizer(problem)
    x, _, _ = opt.run()
    print("x:", x)

if __name__ == "__main__":
    two_nonlinear_springs_opt()






