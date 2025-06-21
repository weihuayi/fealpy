from .. import logger

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import (eigs, cg,  dsolve,  gmres, lgmres,
                                 LinearOperator, spsolve_triangular)
#try:
#    from pypardiso import spsolve
#except ImportError:
#    logger.error("Can't import spsolve from pypardiso! Please install it by `pip install -U pypardiso`.")

from .amg_coarsen import ruge_stuben_chen_coarsen 
from .amg_interpolation import two_points_interpolation
from ..decorator import timer

def spsolve(A, b):
    from mumps import DMumpsContext
    from scipy.sparse.linalg import minres, gmres

    NN = len(b)
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    x = np.array(b)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy() # Cleanup
    return x

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))

class GAMGSolver():
    """
    @brief 几何与代数多重网格的快速解法器

    @note 
    1. 多重网格方法通常分为几何和代数两种类型 
    2. 多重网格方法用到两种插值算子：延拓（Prolongation）和 限制（Restriction）算子
    3. 延拓是指把粗空间上的解插值到细空间中
    4. 限制是指把细空间上的解插值到粗空间中
    5. 几何多重网格利用网格的几何结构来构造延拓和限制算子
    6. 代数多重网格得用离散矩阵的结构来构造延拓和限制算子
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

    @timer
    def setup(self, A, space=None, cdegree=[1]):
        """
        @brief 给定离散矩阵 A, 构造从细空间到粗空间的插值算子

        @param[in] A 矩阵
        @param[in] space 离散空间
        @param[in] cdegree 粗空间的次数

        @note 注意这里假定第 0 层为最细层，第 1、2、3 ... 层变的越来越粗
        """

        # 1. 建立初步的算子存储结构
        self.A = [A]
        self.L = [ ] # 下三角 
        self.U = [ ] # 上三角
        self.D = [ ] # 对角线
        self.P = [ ] # 延拓算子
        self.R = [ ] # 限制矩阵

        # 2. 高次元空间到低次元空间的粗化
        if space is not None:
            Ps = space.prolongation_matrix(cdegree=cdegree)
            for P in Ps:
                self.L.append(sp.tril(self.A[-1], format='csr'))
                self.U.append(sp.triu(self.A[-1], format='csr'))
                self.D.append(self.A[-1].diagonal())
                self.P.append(P)
                R = P.T.tocsr()
                self.R.append(R)
                self.A.append(R @ self.A[-1] @ P)

        # 3. 基于矩阵或几何信息的粗化, 目前只有代数的粗化
        # TODO：增加几何粗化处理 
        NN = np.ceil(np.log2(self.A[-1].shape[0])/2-4)
        NL = max(min( int(NN), 8), 2) # 估计粗化的层数 
        for l in range(NL):
            self.L.append(sp.tril(self.A[-1]).tocsr()) # 前磨光的光滑子
            self.U.append(sp.triu(self.A[-1]).tocsr()) # 后磨光的光滑子
            self.D.append(self.A[-1].diagonal())

            isC, G = ruge_stuben_chen_coarsen(self.A[-1], self.theta)
            P, R = two_points_interpolation(G, isC)
            self.P.append(P)
            self.R.append(R)

            self.A.append((R@self.A[-1]@P).tocsr())
            if self.A[-1].shape[0] < self.csize:
                break

        # 计算最粗矩阵最大和最小特征值
        A = self.A[-1].toarray()
        emax, _ = eigs(A, 1, which='LM')
        emin, _ = eigs(A, 1, which='SM')

        # 计算条件数的估计值
        condest = abs(emax[0] / emin[0])

        if condest > 1e12:
            N = self.A[-1].shape[0]
            self.A[-1] += 1e-12*sp.eye(N)  

    def construct_coarse_equation(self, A, F, level=1):
        """
        @brief 给定一个线性代数系统，利用已经有的延拓和限制算子，构造一个小规模
               的问题
        """
        for i in range(level):
            A = (self.R[i] @ A @ self.P[i]).tocsr()
            F = self.R[i]@F

        return A, F

    def prolongate(self, uh, level): 
        """
        @brief 给定一个第 level 层的向量，延拓到最细层
        """
        assert level >= 1
        for i in range(level-1, -1, -1):
            uh = self.P[i]@uh
        return uh

    def restrict(self, uh, level):
        """
        @brief 给定一个最细层的向量，限制到第 level 层
        """
        for i in range(level):
            uh = self.R[i]@uh
        return uh


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

    @timer
    def solve(self, b):
        """
        @brief 用多重网格方法求解 Ax = b
        """
        N = self.A[0].shape[0]

        if self.ptype == 'V':
            P = LinearOperator((N, N), matvec=self.vcycle, dtype=self.A[0].dtype)
        elif self.ptype == 'W':
            P = LinearOperator((N, N), matvec=self.wcycle, dtype=self.A[0].dtype)
        elif self.ptype == 'F':
            P = LinearOperator((N, N), matvec=self.fcycle, dtype=self.A[0].dtype)

        if self.isolver == 'CG':
            counter = IterationCounter()
            x, info = cg(self.A[0], b, M=P, tol=self.rtol, atol=self.atol, callback=counter)
            print(info)

        return x


    def vcycle(self, r, level=0):
        """
        @brief V-Cycle 方法求解 Ae=r  

        @note
        1. 先在最细空间上进行几次（通常为1到2次）光滑（即迭代求解）操作，这个步骤称为前磨光, 它可以消除高频误差。
        2. 计算残差并将其限制到下一更粗的空间中
        3. 在更粗的空间上，对该残差方程进行几次（通常为1到2次）迭代求解。
        4. 重复步骤2和3，直到达到最粗的空间，在最粗网格上通常用直接法直接求解。
        5. 进行延拓操作，将粗空间误差延拓到相邻细空间上，将此解作为相邻细空间问题的初始猜测。
        6. 在每个更细的空间中，先进行后磨光（即再次迭代求解），然后再将解延拓到下一个更细的空间中
        7. 重复步骤6，直到达到最细的网格。
        """

        NL = len(self.A)
        r = [None]*level + [r] # 残量列表
        e = [None]*level       # 误差列表 

        # 前磨光
        for l in range(level, NL - 1, 1):
            el = spsolve(self.L[l], r[l])
            for i in range(self.sstep):
                el += spsolve(self.L[l], r[l] - self.A[l] @ el)
            e.append(el)
            r.append(self.R[l] @ (r[l] - self.A[l] @ el))

        el = spsolve(self.A[-1], r[-1])
        e.append(el)

        # 后磨光
        for l in range(NL - 2, level - 1, -1):
            e[l] += self.P[l] @ e[l + 1]
            e[l] += spsolve(self.U[l], r[l] - self.A[l] @ e[l])
            for i in range(self.sstep): # 后磨光
                e[l] += spsolve(self.U[l], r[l] - self.A[l] @ e[l])

        return e[level]

    def wcycle(self, r, level=0):
        """
        @brief W-Cycle 方法求解 Ae=r

        @param r 第 level 空间层的残量
        @param level 空间层编号
        """

        NL = len(self.A)
        if level == (NL - 1): # 如果是最粗层
            e = spsolve(self.A[-1], r)
            return e

        e = spsolve(self.L[level], r)
        for s in range(self.sstep):
            e += spsolve(self.L[level], r - self.A[level] @ e) 

        rc = self.R[level] @ ( r - self.A[level] @ e) 

        ec = self.wcycle(rc, level=level+1)
        ec += self.wcycle( rc - self.A[level+1] @ ec, level=level+1)
        
        e += self.P[level] @ ec
        e += spsolve(self.U[level], r - self.A[level] @ e)
        for s in range(self.sstep):
            e += spsolve(self.U[level], r - self.A[level] @ e)
        return e


    def fcycle(self, r):
        """
        @brief F-Cycle 方法求解 Ae=r
        """
        NL = len(self.A)
        r = [r] 
        e = [ ]

        # 从最细层到次最粗层
        for l in range(0, NL - 1, 1):
            el = self.vcycle(r[l], level=l)
            for s in range(self.sstep):
                el += self.vcycle(r[l] - self.A[l] @ el, level=l)

            e.append(el)
            r.append(self.R[l] @ (r[l] - self.A[l] @ e[l]))

        # 最粗层直接求解 
        ec = spsolve(self.A[-1], r[-1])
        e.append(ec)

        # 从次最粗层到最细层
        for l in range(NL - 2, -1, -1):
            e[l] += self.P[l] @ e[l+1]
            e[l] += self.vcycle(r[l] - self.A[l] @ e[l], level=l)
            for s in range(self.sstep):
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



        

        
 

        












