import numpy as np

from fealpy.opt import Problem
from scipy.sparse.linalg import LinearOperator

from fealpy.opt.MatrixVectorProductGradientOptimizer import MatrixVectorProductGradientOptimizer

class TwoNonlinearSpingsProblem(Problem):
    def __init__(self):
        x0 = np.zeros(2, dtype=np.float64)
        super().__init__(x0, self.energy)

    def energy(self, x):
        f = np.array([0, 100])
        U = np.array([25*x[0]**2 + 500/3*x[0]**3 + 50*(x[1]-x[0])**2 + 200/3*(x[1]-x[0])**3 - f[1]*x[1]])
        P = np.array([300*x[0]**2+400*x[0]*x[1]-200*x[1]**2+150*x[0]-100*x[1],
                     200*x[0]**2-400*x[0]*x[1]+200*x[1]**2-100*x[0]+100*x[1]])
        return U, P

#    def springs_stiffness_matrix(self, x):
 #       P = np.array([300*x[0]**2+400*x[0]*x[1]-200*x[1]**2+150*x[0]-100*x[1],
  #                   200*x[0]**2-400*x[0]*x[1]+200*x[1]**2-100*x[0]+100*x[1]])
   #     return P

    def tagnent_stiffness_matrix(self, x):
        Kt = np.array([[600*x[0]+400*x[1]+150, 400*(x[0]-x[1])-100],
                       [400*(x[0]-x[1])-100, 400*x[1]-400*x[0]+100]])
        return Kt

    def newton_raphson_preconditioner(self, x):
        f = np.array([0, 100])
        P = self.energy(x)[1]
        R = f - P
        Kt = self.tagnent_stiffness_matrix(x)
        delx = np.linalg.solve(Kt, R)
        x_next = x + delx
        return x_next

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






