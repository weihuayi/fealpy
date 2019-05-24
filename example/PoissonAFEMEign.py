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
2d
example1: 0.2 50 0
Alg_0: eigns 9.640715013424611 time:13.600042952999502
Alg_1: eigns 9.640760840869085 time:14.150334813999507
Alg_2: eigns 9.640680997643182 time:26.747130691000166
Alg_3: eigns 9.640668516602153 time:43.162928011000076
Alg_4: eigns 9.640708671817325 time:23.129021770999316

example3_A: 0.2 30 0
Alg_0: eigns 2.5386481391782 time:6.16895814199961
Alg_1: eigns 2.538643833700688 time:6.252381130999311
Alg_2: eigns 2.53864261267873 time:15.674745033999898
Alg_3: eigns 2.538642308563923 time:16.235309056000006
Alg_4: eigns 2.5386955621286473 time:8.812511997999536

example3_B: 0.2 30 0
Alg_0: eigns 2.538620571497001 time:6.224730527999782
Alg_1: eigns 2.538622423395317 time:6.865756642000633
Alg_2: eigns 2.538621230101104 time:6.953676066000298
Alg_3: eigns 2.5386205105509805 time:17.44484835500043
Alg_4: eigns 2.538756596200568 time:9.6888128119999246


3D

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
pde = EigenHarmonicOscillator3d()
# pde = EigenSchrodinger3d()

model = EllipticEignvalueFEMModel(
        pde,
        theta=theta,
        maxit=maxit,
        step=step,
        n=3,  # 初始网格加密次数
        p=1,  # 线性元
        q=3,  # 积分精度
        resultdir='/home/why/result/',
        sigma=None,
        multieigs=False)

u0 = model.alg_0()
u1 = model.alg_1()
u2 = model.alg_2()
u3 = model.alg_3()
u4 = model.alg_4()
model.savesolution(u0, 'u0.mat')
model.savesolution(u1, 'u1.mat')
model.savesolution(u2, 'u2.mat')
model.savesolution(u3, 'u3.mat')
model.savesolution(u4, 'u4.mat')
