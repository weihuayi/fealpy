#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: cross_ipcs.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 18 Apr 2024 10:59:38 AM CST
	@bref 
	@ref 
'''  
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.pde.navier_stokes_equation_2d import Poisuille
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.cfd import NSFEMSolver
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DirichletBC
from scipy.sparse.linalg import spsolve

## 参数
rho_g = 
