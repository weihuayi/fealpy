#!/usr/bin/env python3
# 

import sys
from fealpy.pde.EigenvalueData2d import EigenLShape2d
from fealpy.pde.EigenvalueData2d import EigenSquareDC
from fealpy.pde.EigenvalueData2d import EigenCrack
from fealpy.pde.EigenvalueData2d import EigenGWWA, EigenGWWB

from fealpy.pde.EigenvalueData3d import EigenLShape3d
from fealpy.pde.EigenvalueData3d import EigenHarmonicOscillator3d
from fealpy.pde.EigenvalueData3d import EigenSchrodinger3d


from fealpy.fem import EllipticEignvalueFEMModel

"""
几种计算最小特征值最小特征值方法的比较


Usage
-----

./PoissonAFEMEign.py 0.2 50 0

Test Environment
----------------
System: Ubuntu 18.04.2 LTS 64 bit
Memory: 15.6 GB
Processor: Intel® Core™ i7-4702HQ CPU @ 2.20GHz × 8

Result
------

"""

print(__doc__)

theta = float(sys.argv[1])
maxit = int(sys.argv[2])
step = int(sys.argv[3])

info = """
theta : %f
maxit : %d
 step : %d
""" % (theta, maxit, step)
print(info)

# pde = EigenLShape2d()
# pde = EigenSquareDC()
# pde = EigenCrack()

# pde = EigenGWWA()
# pde = EigenGWWB()

# pde = EigenLShape3d()
# pde = EigenHarmonicOscillator3d()
pde = EigenSchrodinger3d()

model = EllipticEignvalueFEMModel(
        pde,
        theta=theta,
        maxit=maxit,
        step=step,
        n=6,  # 初始网格加密次数
        p=1,  # 线性元
        q=3,  # 积分精度
        resultdir='/home/why/result/',
        sigma=None,
        multieigs=False)

u1 = model.alg_3_1()
model.savesolution(u1, 'u1.mat')

u2 = model.alg_3_2()
model.savesolution(u2, 'u2.mat')

u3 = model.alg_3_3()
model.savesolution(u3, 'u3.mat')

u4 = model.alg_3_4()
model.savesolution(u4, 'u4.mat')
