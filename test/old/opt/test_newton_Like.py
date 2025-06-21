import numpy as np
from fealpy.opt import Problem,NewtonRaphsonOptimizer,PLBFGS
from scipy.sparse.linalg import LinearOperator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class RosenbrockProblem(Problem):
    '''
    f(x) = 100(x_2-x_1^2)^2+(1-x_1)^2
    '''
    def __init__(self,x0):
        super().__init__(x0,self.rosenbrockfunction)

    def rosenbrockfunction(self,x):
        f = 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1-x[0])*(1-x[0])
        df = self.grad(x)
        return f,df

    def grad(self,x):
        df = np.zeros(2)
        df[0] = -400*(x[1]-x[0]*x[0])*x[0]-2*(1-x[1])
        df[1] = 200*(x[1]-x[0]*x[0])
        return df

    def Hessian(self,x):
        H = np.zeros((2,2))
        H[0,0] = 1200*x[0]*x[0]-400*x[1]+2
        H[0,1] = -400*x[0]
        H[1,0] = H[0,1]
        H[1,1] = 200
        return H

    def Modified_Hessian(self,x):
        H = np.zeros((2,2))
        H[0,0] = 1200*x[0]*x[0]-400*x[1]+2
        H[0,1] = -400*x[0]
        H[1,0] = H[0,1]
        H[1,1] = 200
        H = self.Cholesky_add_Identity(H)
        return H

    def Modified_Hessian2(self,x):
        H = np.zeros((2,2))
        H[0,0] = 1200*x[0]*x[0]-400*x[1]+2
        H[0,1] = -400*x[0]
        H[1,0] = H[0,1]
        H[1,1] = 200
        H = self.Modified_Cholesky(H)
        return H
  
    def Cholesky_add_Identity(self,A):
        beta = 1e-3
        d = np.diagonal(A)
        if np.min(d)>0:
            tau=0
        else:
            tau = -np.min(d)+beta
        I = np.diag(np.ones(A.shape[0]))
        H = A+tau*I
        while True:
            try:
                np.linalg.cholesky(H)
                return H
            except np.linalg.LinAlgError:
                tau = max(2*tau,beta)
                H = A+tau*I

    def Modified_Cholesky(self,A):
        gamma = np.max(np.abs(np.diagonal(A)))
        xi = np.max(np.abs(np.triu(A)+np.tril(A)))
        n = A.shape[0]
        beta2 = max(gamma,xi/np.sqrt(n*n-1),1e-16)
        delta = 1e-3
        L = np.eye(A.shape[0])
        D = np.zeros(n)
        D[0] = A[0,0]
        for i in range(1,n):
            c = np.zeros(i+1)
            for j in range(i+1):    
                if j==i:
                    c[i] = A[i,i]-np.sum(D[:i]*L[i,:i]**2)
                    D[i] = max(np.abs(c[i]),max(np.abs(c[:i])**2/beta2),delta)
                else:
                    c[j] = A[i,j]-np.sum(D[:j-1]*L[j,:j-1]*L[i,:j-1])
                    L[i,j] = c[j]/D[j]
        return L@np.diag(D)@L.T

    def Newton_preconditioner(self,x):
        H = self.Hessian(x)
        H_inv = np.linalg.inv(H)
        return H_inv
    
    def Modified_Newton_preconditioner(self,x):
        H = self.Modified_Hessian(x)
        H_inv = np.linalg.inv(H)
        return H_inv
     
    def Modified_Newton_preconditioner2(self,x):
        H = self.Modified_Hessian2(x)
        H_inv = np.linalg.inv(H)
        return H_inv

def test_newton(x1):
    x = x1.copy()
    problem = RosenbrockProblem(x)
    problem.MaxIters = 100
    #problem.FunValDiff = 1e-10
    problem.Linesearch = 'quadratic'
    NDof = len(problem.x0)
    problem.Preconditioner = problem.Newton_preconditioner
    opt = NewtonRaphsonOptimizer(problem)
    x,f,gradf = opt.run()
    print(x)
    print(f)

def test_modified_newton(x1):
    x = x1.copy()
    problem = RosenbrockProblem(x)
    problem.MaxIters = 100
    problem.Linesearch = 'wolfe'
    NDof = len(problem.x0)
    problem.Preconditioner = problem.Modified_Newton_preconditioner
    opt = NewtonRaphsonOptimizer(problem)
    x,f,gradf = opt.run()
    print(x)
    print(f)

def test_modified_newton2(x1):
    x = x1.copy()
    problem = RosenbrockProblem(x)
    problem.MaxIters = 100
    problem.Linesearch = 'wolfe'
    NDof = len(problem.x0)
    problem.Preconditioner = problem.Modified_Newton_preconditioner2
    opt = NewtonRaphsonOptimizer(problem)
    x,f,gradf = opt.run()
    print(x)
    print(f)

def test_lbfgs_newton(x1):
    x = x1.copy()
    problem = RosenbrockProblem(x)
    problem.MaxIters = 100
    NDof = len(problem.x0)
    opt = PLBFGS(problem)
    x,f,gradf,_ = opt.run()
    print(x)
    print(f)

def show_function():
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    x0 = np.arange(-2,2,0.02)
    x1 = np.arange(-2,2,0.02)
    X0,X1 = np.meshgrid(x0,x1)
    Z = 100*(X1-X0*X0)*(X1-X0*X0)+(1-X0)*(1-X0)
    ax3.plot_surface(X0,X1,Z,cmap='rainbow')
    plt.show()

if __name__=='__main__': 
    x1 = np.array([1.2,1.2])
    print("牛顿法：")
    test_newton(x1)
    print("加上单位矩阵的修正牛顿法：")
    test_modified_newton(x1)
    print("改进Cholesky分解的修正牛顿法：")
    test_modified_newton2(x1)
    print("L-BFGS拟牛顿法：")
    test_lbfgs_newton(x1)
    #show_function()
       
