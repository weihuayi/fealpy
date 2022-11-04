#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.geometry import CuboidDomain
from fealpy.geometry import SphereDomain
from fealpy.geometry import CylinderDomain

from fealpy.geometry import dunion
from fealpy.geometry import huniform
from fealpy.mesh import DistMesher3d 
from fealpy.mesh import MeshFactory as MF


parser = argparse.ArgumentParser(description=
        """
        DisMesher3d 算法生成四面体网格。
        """)

parser.add_argument('--domain', 
        default=0, type=int, 
        help=
        """
        指定要运行的例子\n
        0 : 单位球体 \n
        1 : 圆柱体\n
        """)

parser.add_argument('--hmin', 
        default=0.1, type=float, 
        help="最小网格尺寸值，默认 0.1")

parser.add_argument('--hmax', 
        default=0.1, type=float, 
        help="最大网格尺寸值，默认与最小网格尺寸相同")

parser.add_argument('--maxit', 
        default=500, type=int, 
        help="最大迭代次数，默认 500 次")

args = parser.parse_args()
domain = args.domain
hmin = args.hmin
hmax = args.hmax
maxit = args.maxit

if domain == 0: # 立方体 

    def fh(p, *args):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.sqrt(x**2 + y**2 + z**2)
        h = hmin + d*0.1
        h[h>hmax] = hmax 
        return h

    domain = CuboidDomain(fh=fh)

elif domain == 1: # 球体 
    def fh(p, *args):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.sqrt(x**2 + y**2 + z**2)
        h = hmin + d*0.1
        h[h>hmax] = hmax 
        return h

    domain = SphereDomain(fh=fh)

elif domain == 2: # 圆柱
    def fh(p, *args):
        d = np.abs(p[:, 2])
        h = hmin + d*0.1
        h[h>hmax] = hmax 
        return h
    domain = CylinderDomain(fh=fh)

elif domain == 3:
    h = 1.0
    r = 0.5

    c0 = np.array([0.0, 0.0, 0.5])
    d0 = CylinderDomain(c0, h, r)
    c1 = np.array([0.0, 0.0, -0.5])
    d1 = CylinderDomain(c1, h, r)

    class TwoCylinderDomain():
        def __init__(self, d0, d1, fh=huniform):
            self.d0 = d0
            self.d1 = d1
            b0 = d0.box
            b1 = d1.box
            self.box = [ 
                    np.minimum(b0[0], b1[0]), np.maximum(b0[1], b1[1]),
                    np.minimum(b0[2], b1[2]), np.maximum(b0[3], b1[3]),
                    np.minimum(b0[4], b1[4]), np.maximum(b0[5], b1[5])
                    ]
            self.fh = fh
            self.facets = {0:None, 1:None}

        def __call__(self, p):
            return dunion(self.d0(p), self.d1(p))

        def signed_dist_function(self, p):
            return self(p)

        def sizing_function(self, p):
            return self.fh(p, self)

        def facet(self, dim):
            return self.facets[dim]

    def fh(p, *args):
        d = np.abs(p[:, 2])
        h = hmin + d*0.1
        h[h>hmax] = hmax 
        return h
    domain = TwoCylinderDomain(d0, d1, fh=fh)

elif domain == 4: # 偶极子天线模型
    l = 4 # 波长参数 [m]
    a = 1 # 偶极天线臂长 [m]
    r = 0.05 # 天线臂半径 [m]
    g = 0.01 # 两臂之间的 Gap 宽度 [m]
    h = 2*a + g # 整个圆柱的长度 
    c = np.array([0.0, 0.0, 0.0]) # 圆柱中心点
    d = np.array([0.0, 0.0, 1.0]) # 圆柱中心线方向
    
    hmin = g/2
    n = np.ceil(2*np.pi*r/hmin)
    dt = 2*np.pi/n
    theta  = np.arange(0, 2*np.pi, dt)

elif domain == 5:
    domain = TorusDomain()
    
mesher = DistMesher3d(domain, hmin, output=True)
mesh = mesher.meshing(maxit)
mesh.to_vtk(fname='test.vtu')
