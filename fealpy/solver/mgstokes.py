    
from ..backend import bm
from ..decorator import cartesian
from .. import logger
from ..solver import spsolve

from scipy.sparse.linalg import spsolve_triangular

import time

class MGStokes():
    def __init__(self, Ai, Bi, Bti, bigAi, P_u, R_u, P_p, R_p, 
                    Nu, Np, level, auxMat, options):
        self.SGS_time = 0
        self.MUL_time = 0

        self.coarse_time = 0
        self.smoothing_time = 0
        self.coarse_count = 0
        self.smoothing_count = 0
        self.cycle_MUL_time = 0

        self.Ai = Ai
        self.Bi = Bi
        self.Bti = Bti
        self.bigAi = bigAi

        self.P_u = P_u
        self.R_u = R_u
        self.P_p = P_p
        self.R_p = R_p

        self.Nu = Nu
        self.Np = Np

        self.level = level
        self.auxMat = auxMat

        self.eps = 1e-10
        self.level = options.get('level', 4)

        self.options = options
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  
        self.maxIt = options.get('solvermaxit', 200)  
        self.N0 = options.get('N0', 500)
        self.solver = options.get('solver', 'direct')

        self.cycle_type = options.get('cycle_type', 'VCYCLE')
        self.smoothing_times = options.get('smoothing_times', 1)

        self.options = options


    def vcycle(self, ru, rp, J=None):
        if J is None:
            J = self.level - 1
        if J == 0:
            start = time.time()
            r = bm.concat([ru, rp], axis=0)
            n = len(rp)
            e = spsolve(self.bigAi, r)
            self.coarse_count += 1
            self.coarse_time += time.time() - start
            return e[:-n], e[-n:]
        
        P_u = self.P_u[J-1]
        P_p = self.P_p[J-1] 
        R_u = self.R_u[J-1]
        R_p = self.R_p[J-1] 

        # pre-smoothing
        eu, ep = self.smoothing(bm.zeros((3*self.Nu[J],), dtype=bm.float64),
                                bm.zeros((self.Np[J],), dtype=bm.float64),ru,rp,J)
        if self.smoothing_times > 1:
            for _ in range(self.smoothing_times-1):
                eu, ep = self.smoothing(eu,ep,ru,rp,J)

        # form residual and restrict onto coarse grid
        start = time.time()
        rru = ru - self.Ai[J] @ eu - self.Bti[J] @ ep
        rrp = rp - self.Bi[J] @ eu

        ruc = R_u @ rru
        rpc = R_p @ rrp
        self.cycle_MUL_time += time.time() - start
        # coarse grid correction
        euc, epc = self.vcycle(ruc, rpc, J-1)

        # correction on the fine grid
        start = time.time()
        tempeu = P_u @ euc
        tempep = P_p @ epc
        self.cycle_MUL_time += time.time() - start
        eu += tempeu
        ep += tempep

        # post-smoothing
        for _ in range(self.smoothing_times):
            eu, ep = self.smoothing(eu,ep,ru,rp,J)
        return eu, ep   

    def wcycle(self, r, J=None): 
        pass       
    
    def smoothing(self, u, p, f, g, J):
        """Solve LUe = r.
        """
        auxMat = self.auxMat[J]
        smootherOpt = self.options
        A = self.Ai[J]
        B = self.Bi[J]
        start = time.time()
        smoother = StokesLSCDGS(auxMat,smootherOpt)
        u, p, self.SGS_time, self.MUL_time = smoother.run(u,p,f,g,A,B,self.SGS_time,self.MUL_time)
        t = time.time() - start
        self.smoothing_time += t
        self.smoothing_count += 1
        return u, p    

    def solve(self, A, F):
        # initial set up
        bigu = bm.zeros_like(F)
        bigr = F

        k = 0
        nb = bm.linalg.norm(F)
        err = bm.zeros((self.maxIt, 1), dtype=bm.float64)
        err[0] = bm.linalg.norm(bigr) / nb
        logger.info(f'Step 5. 进入主循环迭代\n')
        start = time.time()
        while (bm.max(err[k]) > self.tol) & (k <= self.maxIt):
            k = k + 1
            pdof = self.Np[-1]
            if self.cycle_type == 'VCYCLE':
                eu, ep = self.vcycle(bigr[:-pdof], bigr[-pdof:])
            elif self.cycle_type == 'WCYCLE':
                eu, ep = self.wcycle(bigr[:-pdof], bigr[-pdof:])
            
            bigerru = bm.concat([eu, ep])
            bigu = bigu + bigerru
            bigr = bigr - A @ bigerru

            # compute the relative error
            err[k] = bm.linalg.norm(bigr) / nb

            print(
                f'MG Vcycle iter: {k:2d},   '
                f'err = {bm.max(err[k, :]):8.4e}\n'
            )

        self.auxMat
        err = err[:k]
        itStep = k
        cost = time.time() - start
        logger.info(f'Step 6. 程序结束, 开始输出打印结果\n')
        # Output
        print(f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"level = {self.level},   "
            f"total time: {cost}\n\n"
            f"粗网格上求解次数: {self.coarse_count}\n"
            f"粗网格总时间占比: {self.coarse_time / cost},  \n"
            f"Smoothing总时间占比: {self.smoothing_time / cost},   \n"
            f"粗网格和平滑总时间占比: {(self.coarse_time+self.smoothing_time) / cost},   \n"
            )
        
        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return bigu


class StokesLSCDGS():
    def __init__(self, 
        auxMat,
        smootherOpt
    ):  
        self.set_up(auxMat, smootherOpt)
    
    def set_up(self, auxMat, smootherOpt):
        self.smoothingstep = smootherOpt.get('smoothingstep', 2)
        self.smoothingSp = smootherOpt.get('smoothingSp', 'SGS')
        self.smoothingbarSp = smootherOpt.get('smoothingbarSp', 'SGS')
        self.smoothingbarSpPara = smootherOpt.get('smoothingbarSpPara', 1.3)
        if (self.smoothingbarSp == 'VCYCLE') or (self.smoothingSp == 'VCYCLE'):
            self.optionmg = {
                'solvermaxit': 1,
                'solver': 'VCYCLE',
                'smoothingstep': 2,
                'printlevel': 0,
                'setupflag': 0
            }
        
        self.Bt = auxMat.get('Bt')
        # self.BBt = auxMat.get('BBt')
        self.BABt = auxMat.get('BABt')
        # self.Su = auxMat.get('Su')
        self.Su0 = auxMat.get('Su0')
        self.Sp = auxMat.get('Sp')
        self.Spt = auxMat.get('Spt')

        self.invSp = auxMat.get('invSp')
        self.invSpt = auxMat.get('invSpt')
        # self.DSp = auxMat.get('DSp')

        self.n = 0

    def run(self, u,p,f,g,A,B,SGS_time,MUL_time):
        for _ in range(self.smoothingstep):
            # Step 1: relax Momentum eqns
            start = time.time()
            r = (f - self.Bt @ p - A @ u)
            MUL_time += time.time() - start
            start = time.time()

            u += spsolve_triangular(self.Su0, r.reshape(-1,3,order='F')).reshape(-1,order='F')
            SGS_time += time.time() - start
            # Step 2: relax transformed Continuity eqns
            start = time.time()
            rp = g - B @ u
            MUL_time += time.time() - start
            start = time.time()
            
            if self.smoothingSp == 'SGS':
                b0 = spsolve_triangular(self.invSp, rp)
                dq = spsolve_triangular(self.Spt, b0, lower=False)

            elif self.smoothingSp == 'GS':
                dq = spsolve_triangular(self.Sp, rp)
            elif self.smoothingSp == 'VCYCLE':
                pass
            SGS_time += time.time() - start
            
            # Step 3: transform the correction back to the original variables
            start = time.time()
            u = u + self.Bt @ dq
            dq = self.BABt @ dq
            MUL_time += time.time() - start
            
            start = time.time()
            if self.smoothingbarSp == 'SGS':
                b1 = spsolve_triangular(self.invSpt, dq, lower=False)
                dp = spsolve_triangular(self.Sp, b1)
            elif self.smoothingbarSp == 'GS':
                dp = spsolve_triangular(self.Spt, dq, lower=False)
            elif self.smoothingbarSp == 'VCYCLE':
                pass
            SGS_time += time.time() - start

            p = p - self.smoothingbarSpPara*dp
        
        return u, p, SGS_time, MUL_time
    