
#!/usr/bin/env python3
# 
import sys
import argparse
import numpy as np
from fealpy.pde.adi_2d import ADI_2d
from fealpy.mesh import MeshFactory
from fealpy.mesh import TriangleMesh

from numpy.linalg import inv
import matplotlib.pyplot as plt
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.quadrature import  GaussLegendreQuadrature

p=0

##网格
box = [0, 1, 0, 1]
mesh = MeshFactory.boxmesh2d(box, nx=2, ny=2, meshtype='tri') 

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
##
##真解
sigma=3*np.pi
epsilon=1.0
mu=1.0
pde = ADI_2d(sigma,epsilon,mu)
Espace = FirstKindNedelecFiniteElementSpace2d(mesh, p)

M_E=Espace.mass_matrix(epsilon)
M_sig = Espace.mass_matrix(sigma)
M_S=Espace.curl_matrix(1/mu)
# 当前时间步的有限元解Eh
Eh0 = Espace.interpolation(pde.init_E_value)
# 下一层时间步的有限元解
Eh1 = Espace.function()


Hspace=RaviartThomasFiniteElementSpace2d(mesh,p)
# 当前时间步的有限元解Hzh
Hzh0 = Hspace.interpolation(pde.init_H_value)
"""
# 下一层时间步的有限元解
Hzh1 = Hspace.function()
"""

	
	
	


