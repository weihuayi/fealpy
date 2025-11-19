
from ..backend import bm

from scipy.sparse.linalg import spsolve_triangular

from ..solver import spsolve

import time

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
    