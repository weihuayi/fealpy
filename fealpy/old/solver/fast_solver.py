import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, block_diag
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg,  dsolve,  gmres, lgmres, LinearOperator, spsolve_triangular, spsolve
from scipy.sparse.linalg import spilu
from timeit import default_timer as dtimer 

try:
    import pyamg
except ImportError:
    print("请先认真读下面的英文信息！！！")
    print('I do not find  pyamg installed on this system!, so you can not use it.')
    print("""
    If your system is Ubuntu, you can run 

    ```
    $ pip3 install pyamg 
    ```

    If your system is MacOS, you can run

    ```
    $ pip install pyamg
    ```

    If your system is Windows, there are several methods to install `pyamg`

    1. install from conda 
    ```
    conda install -c anaconda pyamg
    ```
    2. Dowload wheel file suitable for your system and python
       from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyamg, then 
       ```
       pip install <pyamg-file-name>.whl
       ```
    """)

from ..decorator import timer

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))

class GaussSeidelSmoother():
    def __init__(self, A):
        """

        Notes
        -----

        对称正定矩阵的 Gauss 光滑

        """

        self.L0 = tril(A).tocsr()
        self.U0 = triu(A, k=1).tocsr()

        self.U1 = self.L0.T.tocsr()
        self.L1 = self.U0.T.tocsr()

    def smooth(self, b, x0, lower=True, maxit=3):
        if lower:
            for i in range(maxit):
                #x0[:] = spsolve_triangular(self.L0, b-self.U0@x0, lower=lower)
                x0[:] = spsolve(self.L0, b-self.U0@x0, permc_spec="NATURAL")
        else:
            for i in range(maxit):
                #x0[:] = spsolve_triangular(self.U1, b-self.L1@x0, lower=lower)
                x0[:] = spsolve(self.U1, b-self.L1@x0, permc_spec="NATURAL")


class JacobiSmoother():
    def __init__(self, A, isDDof=None):
        if isDDof is not None:
            # 处理 D 氏 自由度条件
            gdof = len(isDDof)
            bdIdx = np.zeros(gdof, dtype=np.int_)
            bdIdx[isDDof] = 1 
            Tbd = spdiags(bdIdx, 0, gdof, gdof)
            T = spdiags(1-bdIdx, 0, gdof, gdof)
            A = T@A@T + Tbd

        self.D = A.diagonal() 
        self.L = tril(A, k=-1).tocsr()
        self.U = triu(A, k=1).tocsr()

    def smooth(self, b, maxit=100):
        r = b.copy()
        for i in range(maxit):
            r[:] = b - self.L@r - self.U@r
            r /= self.D
        return r


class HighOrderLagrangeFEMFastSolver():
    def __init__(self, A, F, P, I, isBdDof):
        """


        Notes
        -----
            求解高次拉格朗日有限元的快速算法

            
        """
        self.gdof = len(isBdDof)
        self.A = A # 矩阵 (gdof, gdof), 注意这里是没有处理 D 氏边界的矩阵
        self.F = F # 右端 (gdof, ), 注意这里也没有处理 D 氏边界
        self.I = I # 插值矩阵 (gdof, NN), 把线性元的解插值到 p 次解
        self.isBdDof = isBdDof

        # 获得磨光子
        gdof = self.gdof
        bdIdx = np.zeros(gdof, dtype=np.int_)
        bdIdx[isBdDof] = 1 # 这里假定 A 的前 NN 个自由度是网格节点
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        A = T@A@T + Tbd

        self.L0 = tril(A).tocsr()
        self.U0 = triu(A, k=1).tocsr()

        self.U1 = self.L0.T.tocsr()
        self.L1 = self.U0.T.tocsr()


        # 处理预条件子的边界条件
        NN = P.shape[0]
        bdIdx = np.zeros(NN, dtype=np.int_)
        bdIdx[isBdDof[:NN]] = 1 # 这里假定 A 的前 NN 个自由度是网格节点
        Tbd = spdiags(bdIdx, 0, NN, NN)
        T = spdiags(1-bdIdx, 0, NN, NN)
        P = T@P@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(P)  # P 的 D 氏边界条件用户先处理一下


    def linear_operator(self, b):
        """
        Notes
        -----
        注意这里对 D 氏边界条件的处理与传统的不一样，这里处理的是向量，而不是矩
        阵， 这种处理方法不会改变矩阵的结构。
        """
        isBdDof = self.isBdDof
        r = b.copy()
        val = r[isBdDof]
        r[isBdDof] = 0.0
        r[:] = self.A@r
        r[isBdDof] = val
        return r

    def preconditioner(self, b):
        b = self.smooth(b, lower=True, m=3)
        b = self.I.T@b
        b = self.ml.solve(b, tol=1e-8, accel='cg')       
        b = self.I@b
        b = self.smooth(self.I@b, lower=False, m=3)
        return b

    def smooth(self, b, lower=True, m=3):
        r = np.zeros_like(b)
        if lower:
            for i in range(m):
                r[:] = spsolve_triangular(self.L0, b-self.U0@r, lower=lower)
        else:
            for i in range(m):
                r[:] = spsolve_triangular(self.U1, b-self.L1@r, lower=lower)
        return r

    @timer
    def solve(self, uh, F, tol=1e-8):
        """

        Notes
        -----

        uh 是初值, uh[isBdDof] 中的值已经设为 D 氏边界条件的值, uh[~isBdDof]==0.0
        """

        gdof = self.gdof

        # 处理 Dirichlet 右端边界条件
        isBdDof = self.isBdDof
        F -= self.A@uh
        F[isBdDof] = uh[isBdDof]

        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.preconditioner)
                
        counter = IterationCounter()
        uh[:], info = cg(A, F, M=P, tol=tol, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of cg:", counter.niter)

        return uh 


class LinearElasticityLFEMFastSolver():
    def __init__(self, A, P, isBdDof):
        """
        Notes
        -----
        A: [[A00, A01], [A10, A11]] (2*gdof, 2*gdof)
        
           [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]] (3*gdof, 3*gdof)
        P: 预条件子 (gdof, gdof)

        这里的边界条件处理放到矩阵和向量的乘积运算当中, 所心不需要修改矩阵本身
        """
        self.GD = len(A) 
        self.gdof = P.shape[0]

        self.A = A
        self.isBdDof = isBdDof

        # 处理预条件子的边界条件
        bdIdx = np.zeros(P.shape[0], dtype=np.int_)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, P.shape[0], P.shape[0])
        T = spdiags(1-bdIdx, 0, P.shape[0], P.shape[0])
        P = T@P@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(P) 

    def linear_operator(self, b):
        """
        Notes
        -----
        b: (2*gdof, )
        """
        GD = self.GD
        isBdDof = self.isBdDof
        b = b.copy()
        b = b.reshape(GD, -1)
        val = b[:, isBdDof]
        b[:, isBdDof] = 0.0
        r = np.zeros_like(b)
        for i in range(GD):
            for j in range(GD):
                r[i] += self.A[i][j]@b[j]
        r[:, isBdDof] = val
        return r.reshape(-1)

    def preconditioner(self, b):
        GD = self.GD
        b = b.reshape(GD, -1)
        r = np.zeros_like(b)
        for i in range(GD):
            r[i] = self.ml.solve(b[i], tol=1e-8, accel='cg')       
        return r.reshape(-1)

    @timer
    def solve(self, uh, F, tol=1e-8):
        """

        Notes
        -----

        uh 是初值, uh[isBdDof] 中的值已经设为 D 氏边界条件的值, uh[~isBdDof]==0.0
        """

        GD = self.GD
        gdof = self.gdof

        # 处理 Dirichlet 右端边界条件
        for i in range(GD):
            for j in range(GD):
                F[:, i] -= self.A[i][j]@uh[:, j]
        F[isBdDof] = uh[isBdDof]

        A = LinearOperator((GD*gdof, GD*gdof), matvec=self.linear_operator)
        P = LinearOperator((GD*gdof, GD*gdof), matvec=self.preconditioner)
                
        uh.T.flat, info = cg(A, F.T.flat, M=P, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)

        return uh 

class LinearElasticityLFEMFastSolver_1():
    def __init__(self, A, S, I=None, stype='pamg', drop_tol=None,
            fill_factor=None):
        """

        Notes
        -----

        A : 线弹性矩阵离散矩阵
        S : 刚度矩阵
        I : 刚体运动空间基函数系数矩阵
        """
        self.gdof = S.shape[0] # 标量自由度个数
        self.GD = A.shape[0]//self.gdof
        self.A = A
        self.I = I
        self.stype = stype


        if stype == 'pamg':
            self.smoother = GaussSeidelSmoother(A)
            start = dtimer()
            self.ml = pyamg.ruge_stuben_solver(S) 
            end = dtimer()
            print('time for poisson amg setup:', end - start)
        elif stype == 'lu':
            start = dtimer()
            self.ilu = spilu(A.tocsc(), drop_tol=drop_tol,
                    fill_factor=fill_factor)
            end = dtimer()
            print('time for ILU:', end - start)
        elif stype == 'rm':
            assert I is not None
            self.I = I
            self.AM = inv(I.T@(A@I))
            self.smoother = GaussSeidelSmoother(A)
            start = dtimer()
            self.ml = pyamg.ruge_stuben_solver(S) 
            end = dtimer()
            print('time for poisson amg setup:', end - start)

    def lu_preconditioner(self, r):
        e = self.ilu.solve(r)
        return e

    def pamg_preconditioner(self, r):
        gdof = self.gdof
        GD = self.GD

        e = r.copy() 
        self.smoother.smooth(r, e, lower=True, maxit=3)
        for i in range(GD):
            e[i*gdof:(i+1)*gdof] = self.ml.solve(r[i*gdof:(i+1)*gdof], tol=1e-2)       
        self.smoother.smooth(r, e, lower=False, maxit=3)

        return e 

    def rm_preconditioner(self, r):
        gdof = self.gdof
        GD = self.GD

        e = r.copy()
        self.smoother.smooth(r, e, lower=True, maxit=3)

        rd = r - self.A@e # 更新残量
        for i in range(GD):
            e[i*gdof:(i+1)*gdof] += self.ml.solve(rd[i*gdof:(i+1)*gdof],
                    tol=1e-1)       


        rd = r - self.A@e # 更新残量
        rd = self.I.T@rd
        ed = self.AM@rd
        e += self.I@ed

        rd = r - self.A@e # 更新残量
        for i in range(GD):
            e[i*gdof:(i+1)*gdof] += self.ml.solve(rd[i*gdof:(i+1)*gdof],
                    tol=1e-1)       

        self.smoother.smooth(r, e, lower=False, maxit=3)

        return e


    def solve(self, uh, F, tol=1e-8):
        """

        Notes
        -----
        uh 是初值, uh[isBdDof] 中的值已经设为 D 氏边界条件的值, uh[~isBdDof]==0.0
        """

        GD = self.GD
        gdof = self.gdof
        stype = self.stype

        if stype == 'pamg':
            P = LinearOperator((GD*gdof, GD*gdof), matvec=self.pamg_preconditioner)
        elif stype == 'lu':
            P = LinearOperator((GD*gdof, GD*gdof), matvec=self.lu_preconditioner)
        elif stype == 'rm':
            P = LinearOperator((GD*gdof, GD*gdof), matvec=self.rm_preconditioner)
            
        start = dtimer()

        counter = IterationCounter()
        uh.T.flat, info = cg(self.A, F.T.flat, x0=uh.T.flat, M=P, tol=1e-8,
                callback=counter)
        end = dtimer()
        print('time of pcg:', end - start)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)
        return uh 


class SaddlePointFastSolver():
    def __init__(self, A, F):
        """

        Notes
        -----
            A = (M, B, C), C 可以是 None
            F = (F0, F1), 

            求解如下离散代数系统 
            M   x0 + B x1 = F0 
            B^T x0 + C x1 = F1

        TODO:

        """
        self.A = A
        self.F = F

        M = A[0]
        B = A[1]
        C = A[2]

        self.D = 1.0/M.diagonal() # M 矩阵的对角线的逆
        # S 相当于间断元的刚度矩阵
        S = (B.T@spdiags(self.D, 0, M.shape[0], M.shape[1])@B).tocsr()
        self.ml = pyamg.ruge_stuben_solver(S) # 这里要求必须有网格内部节点 

        # TODO：把间断元插值到连续元线性元空间，然后再做 AMG

    def linear_operator(self, b):
        M = self.A[0]
        B = self.A[1]
        m = M.shape[0]
        n = B.shape[1]
        r = np.zeros_like(b)
        r[:m] = M@b[:m] + B@b[m:]
        r[m:] = B.T@b[:m]
        return r

    def diag_preconditioner(self, b):
        D = self.D
        m = self.A[0].shape[0]
        n = self.A[1].shape[1]

        r = np.zeros_like(b)

        b0 = b[:m]
        b1 = b[m:]

        r[:m] = b0*D
        r[m:] = self.ml.solve(b1, tol=1e-8, accel='cg')       
        return r 
    
    @timer
    def solve(self, tol=1e-8):
        M = self.A[0]
        B = self.A[1]
        C = self.A[2]

        m = M.shape[0]
        n = B.shape[1]
        gdof = m + n

        counter = IterationCounter()
        F = np.r_[self.F[0], self.F[1]]
        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.diag_preconditioner)
        x, info = lgmres(A, F, M=P, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)

        return x[:m], x[m:] 




class LinearElasticityHZFEMFastSolve():
    def __init__(self,A,F,vspace):
        '''

        Notes
        -----
            求解胡张元形成的线弹性力学方程
            A = (M, B)
            F = (F0,F1) 

            离散的代数系统如下
            M x0 + B^T x1 = F0
            B x0          = F1

        '''

        self.M = A[0]
        self.B = A[1]
        self.F = np.r_[F[0],F[1].T.reshape(-1)]
        self.D = self.M.diagonal()
        tgdof = self.M.shape[0]
        mesh = vspace.mesh

        # S 相当于间断元的刚度矩阵
        S = self.B@spdiags(1/self.D,0,tgdof,tgdof)@self.B.T
        self.smoother = GaussSeidelSmoother(S)
        self.S = S


        # construct amg solver
        from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace
        cspace = LagrangeFiniteElementSpace(mesh,1)
        S_coarse = cspace.stiff_matrix(isDDof=cspace.is_boundary_dof()) #粗空间S矩阵的逼近
        #self.ml = pyamg.ruge_stuben_solver(S_coarse) # 这里要求必须有网格内部节点 
        self.ml = pyamg.smoothed_aggregation_solver(S_coarse)
        # Get interpolation matrix
        NC = mesh.number_of_cells()
        bc = vspace.dof.multiIndex/vspace.p #(fldof,gdim+1)
        val = np.tile(bc, (NC, 1)) #(NC*fldof,gdim+1)

        gdim = mesh.geo_dimension()
        fldof = vspace.number_of_local_dofs() #f表示细空间
        cldof = cspace.number_of_local_dofs() #c表示粗空间
        fgdof = vspace.number_of_global_dofs()
        cgdof = cspace.number_of_global_dofs()

        I = np.broadcast_to(vspace.cell_to_dof()[:,:,None],shape=(NC,fldof,gdim+1))
        J = np.broadcast_to(cspace.cell_to_dof()[:,None,:],shape=(NC,fldof,cldof))

        self.PI = csr_matrix((val.flat, (I.flat, J.flat)), shape=(fgdof, cgdof))
        self.vgdof = fgdof*gdim
        self.tgdof = tgdof
        self.gdim = gdim

    def linear_operator(self,b):
        m = self.tgdof
        r = np.zeros_like(b)
        r[:m] = self.M@b[:m]+self.B.T@b[m:]
        r[m:] = self.B@b[:m]
        return r

    def precondieitoner(self,r):
        tgdof = self.tgdof
        gdim = self.gdim
        r1 = r[tgdof:]

        u0 = r[:tgdof]/self.D
        u1 = np.zeros_like(r1)
        r1 -= self.B@u0

        self.smoother.smooth(r1,u1,maxit=10)

        r2 = r1 - self.S@u1

        for i in range(gdim):
            u1[i::gdim] += self.PI@self.ml.solve(self.PI.T@r2[i::gdim],tol=1e-8, accel='cg')
        
        self.smoother.smooth(r1,u1,lower=False,maxit=10)

        return np.r_[u0+self.B.T@u1/self.D, -u1]
        

    @timer
    def solve(self, tol=1e-8):
        m = self.tgdof
        n = self.vgdof
        gdof = m + n

        counter = IterationCounter(disp=False)
        F = self.F
        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.precondieitoner)
        x, info = lgmres(A, F, M=P, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)

        return x


class LevelSetFEMFastSolver():
    def __init__(self, A):
        self.A = A

    @timer
    def solve(self, b, tol=1e-8):

        counter = IterationCounter(disp=False)
        x, info = lgmres(self.A, b, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)

        return x
