#!/usr/bin/env python3
# 

import sys
from fealpy.pde.poisson_2d import LShapeRSinData
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

CrackData:

old method: 24452 nodes
    smallest eigns: 16.746135567145902 with time:  57.65254141800051

hu method: 26265 nodes
    smallest eigns: 16.745923914871994 with time:  14.745063177000702

me method: 24379 nodes
    smallest eigns: 16.746147214383882 with time:  14.566903421999996

LShapeData:
     ./PoissonAFEMEign.py 0.15 50 0

     theta = 0.15
     maxit = 50

    old method: 30148 nodes
        smallest eigns: 9.64071499801269 with time:  51.8588675280007

    hu method: 30158 nodes
        smallest eigns: 9.640716887235763 with time:  16.846451205999983

    me method: 30210 nodes
        smallest eigns: 9.640713070938679 with time:  17.15686360500058
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


pde = LShapeRSinData()

model = EllipticEignvalueFEMModel(
        pde,
        theta=theta,
        maxit=maxit,
        step=step,
        n=3,
        p=1,
        q=3)

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

