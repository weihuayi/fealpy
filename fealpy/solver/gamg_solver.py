
from ..backend import backend_manager as bm
from ..operator import LinearOperator
from .cg import cg
from scipy.sparse.linalg import spsolve_triangular,spsolve
from .amg import ruge_stuben_amg
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor
from .. import logger
from ..utils import timer
import time 
from scipy.sparse.linalg import eigs

class GAMGSolver():
    """
    Fast Solvers for Geometric and Algebraic Multigrid Methods

    1.Multigrid methods are typically divided into two types: geometric and algebraic.
    2.Multigrid methods involve two types of interpolation operators: Prolongation and Restriction.
    3.Prolongation refers to interpolating the solution from a coarse space to a fine space.
    4.Restriction refers to interpolating the solution from a fine space to a coarse space.
    5.Geometric multigrid methods use the geometric structure of the mesh to construct prolongation and restriction operators.
    6.Algebraic multigrid methods use the structure of the discrete matrix to construct prolongation and restriction operators.

    Parameters:
        theta(float) : Coarsening coefficient
        csize(int)   : Size of the coarsest problem
        ctype(str)   : Coarsening method
        itype(str)   : Interpolation method
        ptype(str)   : Preconditioner type
        sstep(int)   : Default number of smoothing steps
        isolver(str) : Default iterative solver; other options include 'MG'
        maxit(int)   : Default maximum number of iterations
        csolver(str) : Default solver for the coarsest grid
        rtol(float)  : Relative error convergence threshold
        atol(float)  : Absolute error convergence threshold

    Returns:
        Tensor: The numerical solutions under Multigrid solver.
    """
    def __init__(self,
            theta: float = 0.025, # 粗化系数
            csize: int = 50, # 最粗问题规模
            ctype: str = 'C', # 粗化方法
            itype: str = 'T', # 插值方法
            ptype: str = 'V', # 预条件类型
            sstep: int = 1, # 默认光滑步数
            isolver: str = 'MG', # 默认迭代解法器，还可以选择'MG'
            maxit: int = 200,   # 默认迭代最大次数
            csolver: str = 'direct', # 默认粗网格解法器
            rtol: float = 1e-8,      # 相对误差收敛阈值
            atol: float = 1e-8,      # 绝对误差收敛阈值
            ):
        self.csize = csize 
        self.theta = theta
        self.ctype = ctype
        self.itype = itype
        self.ptype = ptype
        self.sstep = sstep
        self.isolver = isolver
        self.maxit = maxit
        self.csolver = csolver
        self.rtol = rtol
        self.atol = atol

    def setup(self, A, P=None, R=None, mesh=None, space=None, cdegree=[1]):
        """

        Parameters:
            A (CSRTensor): the matrix
            P (Optional[list]): prolongation matrix, from finnest to coarsest
            R (Optional[list]): restriction matrix, from coarsest to finnest
            mesh (Optional[Mesh]): the mesh
        """
        # 1. Initialize the storage structure for operators
        self.A = [A]  # List to store the system matrices at each level
        self.L = []   # List to store the lower triangular matrices
        self.U = []   # List to store the upper triangular matrices
        self.P = []   # List to store the prolongation operators
        self.R = []   # List to store the restriction matrices

        # 2. Coarsening from higher-order spaces to lower-order spaces.
        if space is not None:
            Ps = space.prolongation_matrix(cdegree=cdegree)
            for p in Ps:
                self.L.append(self.A[-1].tril())
                self.U.append(self.A[-1].triu())
                self.P.append(p)
                r = p.T.tocsr()
                self.R.append(r)
                self.A.append(r @ self.A[-1] @ p)

        if P is not None:
            assert isinstance(P, list)
            self.P += P
            for p in P:
                self.L.append(self.A[-1].tril())
                self.U.append(self.A[-1].triu())
                r = p.T.tocsr()
                self.R.append(r)
                self.A.append(r @ self.A[-1] @ p)
        elif mesh is not None: # geometric coarsening from finnest to coarsest
            pass
        else: # algebraic coarsening 
            NN = bm.ceil(bm.log2(self.A[-1].shape[0])/2-4)
            NL = max(min(int(NN), 8), 2) # 估计粗化的层数 
            start_time = time.time()
            for l in range(NL):
                self.L.append(self.A[-1].tril()) # 前磨光的光滑子
                self.U.append(self.A[-1].triu()) # 后磨光的光滑子

                p, r = ruge_stuben_amg(self.A[-1], self.theta)
                
                self.P.append(p)
                self.R.append(r)

                self.A.append(r @ self.A[-1] @ p)
                if self.A[-1].shape[0] < self.csize:
                    break
            end_time = time.time()
            logger.info(f"Coarsening time: {end_time-start_time}")
            # # 计算最粗矩阵最大和最小特征值
            # A = self.A[-1].toarray()
            # emax, _ = eigs(A, 1, which='LM')
            # emin, _ = eigs(A, 1, which='SM')

            # # 计算条件数的估计值
            # condest = abs(emax[0] / emin[0])

            # if condest > 1e12:
            #     N = self.A[-1].shape[0]
            #     self.A[-1] += 1e-12*bm.eye(N)  

    def construct_coarse_equation(self, A, F, level=1):
        """
        Given a linear algebraic system, construct a smaller-scale problem
        using the existing prolongation and restriction operators.

        Parameters:
            A(CSRTensor): The operator in the fine mesh
            F(Tensor)   : The right vector of the linear system in the fine mesh

        Returns:
            A(CSRTensor): The operator in the coarse mesh
            F(Tensor)   : The right vector of the linear system in the coarse mesh
        """
        for i in range(level):
            A = (self.R[i] @ A @ self.P[i]).tocsr()
            F = self.R[i] @ F

        return A, F

    def prolongate(self, uh, level): 
        """
        Given a vector at level 'level', prolongate it to the finest level.
        
        Parameters:
            uh(Tensor)   : The right vector of the linear system in the coarse of 'level' mesh

        Returns:
            uh(Tensor)   : The right vector of the linear system in the finest mesh
        """
        assert level >= 1
        for i in range(level-1, -1, -1):
            uh = self.P[i] @ uh
        return uh

    def restrict(self, uh, level):
        """
        Given a vector at the finest level, restrict it to level 'level'.

        Parameters:
            uh(Tensor)   : The right vector of the linear system in the finest mesh

        Returns:
            uh(Tensor)   : The right vector of the linear system in the coarse of 'level' mesh
        """
        for i in range(level):
            uh = self.R[i] @ uh
        return uh


    def print(self):
        """
        Print information about the algebraic multigrid.
        """
        NL = len(self.A)
        for l in range(NL):
            print(f"{l}-th level:")
            print(f"A.shape = {self.A[l].shape}")
            if l < NL-1:
                print(f"L.shape = {self.L[l].shape}") 
                print(f"U.shape = {self.U[l].shape}") 
                print(f"D.shape = {self.D[l].shape}")
                print(f"P.shape = {self.P[l].shape}") 
                print(f"R.shape = {self.R[l].shape}")

    def solve(self, b):
        """ 
        Solve Ax=b by gamg method 
        """
        self.kargs = bm.context(b)
        N = self.A[0].shape[0]
        if self.ptype == 'V':
            P = LinearOperator((N, N), matvec = self.vcycle)
        elif self.ptype == 'W':
            P = LinearOperator((N, N), matvec = self.wcycle)
        elif self.ptype == 'F':
            P = LinearOperator((N, N), matvec = self.fcycle)

        if self.isolver == 'CG':
            x0 = bm.zeros(N, **self.kargs)
            x, info = cg(self.A[0], b, x0=x0, M=P, atol=self.atol, rtol=self.rtol, maxit=self.maxit, returninfo=True)
            return x,info
        elif self.isolver == 'MG':
            x0 = bm.zeros(N, **self.kargs)
            x, info = self.mg_solve(b, x0=x0)
            return x,info

    def mg_solve(self,r,x0=None):
        info = {}
        if x0 is not None:
            x = x0
        else:
            x = bm.zeros(r.shape[0],**self.kargs)
        niter = 0
        while True:
            if self.ptype == 'V':
                x += self.vcycle(r-self.A[0] @ x)
            elif self.ptype == 'W':
                x += self.wcycle(r-self.A[0] @ x) 
            elif self.ptype == 'F':
                x += self.fcycle(r-self.A[0] @ x)
            
            niter +=1
            res = r - self.A[0] @ x
            res = bm.linalg.norm(res)
            info['residual'] = res
            info['niter'] = niter
            if res < self.atol:
                logger.info(f"MG: converged in {niter} iterations, "
                            "stopped by absolute tolerance.")
                break

            if res < self.rtol * bm.linalg.norm(r):
                logger.info(f"MG: converged in {niter} iterations, "
                            "stopped by relative tolerance.")
                break

            if (self.maxit is not None) and (niter >= self.maxit):
                logger.info(f"MG: failed, stopped by maxit ({self.maxit}).")
                break

        return x,info
    
    def vcycle(self, r, level=0):
        """
        Solve Ae = r using the V-Cycle method.

        1. Perform a few (typically 1 to 2) smoothing (i.e., iterative solving) operations on the finest space. 
            This step is called pre-smoothing, which helps eliminate high-frequency errors.
        2. Compute the residual and restrict it to the next coarser space.
        3. On the coarser space, perform a few (typically 1 to 2) iterative solves for the residual equation.
        4. Repeat steps 2 and 3 until the coarsest space is reached. 
            On the coarsest grid, a direct solver is usually employed to solve the equation directly.
        5. Perform prolongation, extending the coarse-space error to the adjacent finer space. 
            Use this solution as the initial guess for the finer space problem.
        6. On each finer space, perform post-smoothing (i.e., iterative solving again) 
            before prolongating the solution to the next finer space.
        7. Repeat step 6 until the finest grid is reached.

        Parameters:
            r(Tensor) : The residual in the level-th space.
            level(int): The index of the space level.
        
        Returns:
            Tensor: The solution after one smoothing operation.
        """
        NL = len(self.A)
        r = [None]*level + [r] 
        e = [None]*level       

        # Pre-smoothing
        for l in range(level, NL - 1, 1):
            el = spsolve_triangular(self.L[l].to_scipy(), r[l])
            for i in range(self.sstep):
                el += spsolve_triangular(self.L[l].to_scipy(), r[l] - self.A[l] @ el)
            e.append(el)
            r.append(self.R[l] @ (r[l] - self.A[l] @ el))

        el = spsolve(self.A[-1].to_scipy(), r[-1])
        e.append(el)

        # Post-smoothing
        for l in range(NL - 2, level - 1, -1):
            e[l] += self.P[l] @ e[l + 1]
            e[l] += spsolve_triangular(self.U[l].to_scipy(), r[l] - self.A[l] @ e[l], lower=False)
            for i in range(self.sstep): 
                e[l] += spsolve_triangular(self.U[l].to_scipy(), r[l] - self.A[l] @ e[l], lower=False)

        return e[level]
    
    def wcycle(self, r, level=0):
        """
        Solve Ae = r using the W-Cycle method.

        Parameters:
            r(Tensor) : The residual in the level-th space.
            level(int): The index of the space level.
        
        Returns:
            Tensor: The solution after one smoothing operation.
        """
        NL = len(self.A)
        if level == (NL - 1): 
            e = spsolve(self.A[-1].to_scipy(), r)
            return e

        e = spsolve_triangular(self.L[level].to_scipy(), r)
        for s in range(self.sstep):
            e += spsolve_triangular(self.L[level].to_scipy(), r - self.A[level] @ e) 

        rc = self.R[level] @ ( r - self.A[level] @ e) 

        ec = self.wcycle(rc, level=level+1)
        ec += self.wcycle( rc - self.A[level+1] @ ec, level=level+1)
        
        e += self.P[level] @ ec
        e += spsolve_triangular(self.U[level].to_scipy(), r - self.A[level] @ e, lower=False)
        for s in range(self.sstep):
            e += spsolve_triangular(self.U[level].to_scipy(), r - self.A[level] @ e, lower=False)
        return e
        
    def fcycle(self, r):
        """
        Solve Ae = r using the F-Cycle method.

        Parameters:
            r(Tensor) : The residual in the level-th space.
            level(int): The index of the space level.
        
        Returns:
            Tensor: The solution after one smoothing operation.
        """
        NL = len(self.A)
        r = [r] 
        e = [ ]

        for l in range(0, NL - 1, 1):
            el = self.vcycle(r[l], level=l)
            for s in range(self.sstep):
                el += self.vcycle(r[l] - self.A[l] @ el, level=l)

            e.append(el)
            r.append(self.R[l] @ (r[l] - self.A[l] @ e[l]))

        ec = spsolve(self.A[-1].to_scipy(), r[-1])
        e.append(ec)

        for l in range(NL - 2, -1, -1):
            e[l] += self.P[l] @ e[l+1]
            e[l] += self.vcycle(r[l] - self.A[l] @ e[l], level=l)
            for s in range(self.sstep):
                e[l] += self.vcycle(r[l] - self.A[l] @ e[l], level=l)

        return e[0]
    
    # def bpx(self, r):
    #     """
    #     @brief 
    #     @note
    #     """
    #     NL = len(self.A)
    #     r = [r] 
    #     e = [ ]

    #     for l in range(0, NL - 1, 1):
    #         e.append(r[l]/self.D[l])
    #         r.append(self.R[l] @ r[l])

    #     # 最粗层直接求解 
    #     # TODO: 最粗层增加迭代求解
    #     ec = cg(self.A[-1], r[-1])
    #     e.append(ec)

    #     for l in range(NL - 2, -1, -1):
    #         e[l] += self.P[l] @ e[l+1]

    #     return e[0]
