import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import (eigs, cg,  dsolve,  gmres, lgmres, 
        LinearOperator, spsolve_triangular)
from pypardiso import spsolve
from timeit import default_timer as timer

from .coarsen import ruge_stuben_chen_coarsen 
from .interpolation import standard_interpolation 
from .interpolation import two_points_interpolation

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))

class AMGSolver():
    """
    @brief 代数多重网格解法器类。用代数多重网格方法求解
    @note 该求解器完全移植 iFEM
    """
    def __init__(self,
            theta: float = 0.025, # 粗化系数
            csize: int = 50, # 最粗问题规模
            ctype: str = 'C', # 粗化方法
            itype: str = 'T', # 插值方法
            ptype: str = 'W', # 预条件类型
            sstep: int = 2, # 默认光滑步数
            isolver: str = 'CG', # 默认迭代解法器
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

    def setup(self, A):
        """
        @brief 
        """
        NN = np.ceil(np.log2(A.shape[0])/2-4)
        NL = max(min( int(NN), 8), 2) # 估计粗化的层数 
        self.A = [A]
        self.L = [ ] # 下三角 
        self.U = [ ] # 上三角
        self.D = [ ] # 对角线
        self.P = [ ] # 延长矩阵
        self.R = [ ] # 限止矩阵
        for l in range(NL):
            self.L.append(sp.tril(self.A[l]).tocsr()) # 前磨光的光滑子
            self.U.append(sp.triu(self.A[l]).tocsr()) # 后磨光的光滑子
            self.D.append(self.A[l].diagonal())

            isC, G = ruge_stuben_chen_coarsen(self.A[l], self.theta)
            P, R = standard_interpolation(G, isC)

            print(type(P))
            print(type(R))
            #P, R = two_points_interpolation(G, isC)

            self.A.append((R@self.A[l]@P).tocsr())
            self.P.append(P)
            self.R.append(R)
            if self.A[-1].shape[0] < self.csize:
                break

        # 计算最大和最小特征值
        emax, _ = eigs(self.A[-1], 1, which='LM')
        emin, _ = eigs(self.A[-1], 1, which='SM')

        # 计算条件数的估计值
        condest = abs(emax[0] / emin[0])

        if condest > 1e12:
            N = self.A[-1].shape[0]
            self.A[-1] += 1e-12*sp.eye(N)  

    def print(self):
        """
        @brief 打印代数多重网格的信息
        """
        NL = len(self.A)
        for l in range(NL):
            print(l, "-th level:")
            print("A.shape = ", self.A[l].shape)
            if l < NL-1:
                print("L.shape = ", self.L[l].shape) 
                print("U.shape = ", self.U[l].shape) 
                print("D.shape = ", self.D[l].shape)
                print("P.shape = ", self.P[l].shape) 
                print("R.shape = ", self.R[l].shape) 

    def solve(self, b):
        """
        @brief
        """
        N = self.A[0].shape[0]
        if self.ptype == 'V':
            P = LinearOperator((N, N), matvec=self.vcycle, dtype=self.A[0].dtype)
        elif self.ptype == 'W':
            P = LinearOperator((N, N), matvec=self.wcycle, dtype=self.A[0].dtype)
        if self.isolver == 'CG':
            counter = IterationCounter()
            x, info = cg(self.A[0], b, M=P, tol=self.rtol, atol=self.atol, callback=counter)
            #x, info = cg(self.A[0], b, tol=self.rtol, atol=self.atol, callback=counter)
            print(info)


    def vcycle(self, r, level=0):
        """
        @brief 

        @note 在每一层求解 Ae = r
        """
        NL = len(self.A)
        r = [r] # 残量列表
        e = [ ] # 误差列表 

        for l in range(level, NL - 1, 1):
            #el = spsolve_triangular(self.L[l], r[l], lower=True)
            el = spsolve(self.L[l], r[l])
            for i in range(self.sstep):
                #el += spsolve_triangular(self.L[l], r[l] - self.A[l] @ el, lower=True)
                el += spsolve(self.L[l], r[l] - self.A[l] @ el)
            e.append(el)
            r.append(self.R[l] @ (r[l] - self.A[l] @ el))

        el = spsolve(self.A[-1], r[-1])
        e.append(el)

        for l in range(NL - 2, level - 1, -1):
            e[l] += self.P[l] @ e[l + 1]
            #e[l] += spsolve_triangular(self.U[l], r[l] - self.A[l] @ e[l], lower=False)
            e[l] = spsolve(self.U[l], r[l] - self.A[l] @ e[l])
            for i in range(self.sstep): # 后磨光
                #e[l] += spsolve_triangular(self.U[l], r[l] - self.A[l] @ e[l], lower=False)
                e[l] = spsolve(self.U[l], r[l] - self.A[l] @ e[l])
        return e[0]

    def wcycle(self, r, level=0):
        """

        @param r 第 level 层的残量
        @param level 层编号

        @note 在每一层求解 Ae=r, 其中第 0 层是最细层
        """

        NL = len(self.A)
        if level == (NL - 1): # 如果是最粗层
            e = spsolve(self.A[-1], r)
            return e

        e = spsolve_triangular(self.L[level], r, lower=True)
        for s in range(3):
            e += spsolve_triangular(self.L[level], r - self.A[level] @ e, lower=True) 

        rc = self.R[level] @ ( r - self.A[level] @ e) 

        ec = self.wcycle(rc, level=level+1)
        ec += self.wcycle( rc - self.A[level+1] @ ec, level=level+1)
        
        e += self.P[level] @ ec
        e += spsolve_triangular(self.U[level], r - self.A[level] @ e, lower=False)
        for s in range(3):
            e += spsolve_triangular(self.U[level], r - self.A[level] @ e, lower=False)

        return e


    def fcycle(self, r):
        """
        """
        NL = len(self.A)
        r = [r] 
        e = [ ]

        # 从最细层到次最粗层
        for l in range(0, NL - 1, 1):
            el = self.vcycle(r[l], level=l)
            for s in range(3):
                el += self.vcycle(r[l] - slef.A[l] @ el, level=l)

            e.append(el)
            r.append(self.R[l] @ (r[l] - self.A[l] @ e[l]))

        # 最粗层直接求解 
        ec = spsolve(self.A[-1], r[-1])
        e.append(ec)

        # 从次最粗层到最细层
        for l in range(NL - 2, -1, -1):
            e[l] += self.P[l] @ e[l+1]
            e[l] += self.vcycle(r[l] - self.A[l] @ e[l], level=l)
            for s in range(3):
                e[l] += self.vcycle(r[l] - self.A[l] @ e[l], level=l)

        return e[0]


    def bpx(self, r):
        """
        @brief 
        @note
        """
        NL = len(self.A)
        r = [r] 
        e = [ ]

        for l in range(0, NL - 1, 1):
            e.append(r[l]/self.D[l])
            r.append(self.R[l] @ r[l])

        # 最粗层直接求解 
        # TODO: 最粗层增加迭代求解
        ec = spsolve(self.A[-1], r[-1])
        e.append(ec)

        for l in range(NL - 2, -1, -1):
            e[l] += self.P[l] @ e[l+1]

        return e[0]



        

        
 

        












