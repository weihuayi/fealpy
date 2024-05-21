#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: cross.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 18 May 2024 03:01:14 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from cross_solver import CrossSolver
from cross_pde import CrossWLF
from fealpy.mesh import TriangleMesh
from fealpy.cfd import NSFEMSolver
from fealpy.levelset.ls_fem_solver import LSFEMSolver
from fealpy.functionspace import LagrangeFESpace

ns = 10
T = 10
nt = 500
pde = CrossWLF()
solver  = CrossSolver(pde)
mesh = TriangleMesh.from_box(pde.box, 10*ns, ns)
uspace = LagrangeFESpace(mesh, p=2, doforder='sdofs')
pspace = LagrangeFESpace(mesh, p=1, doforder='sdofs')
phispace = LagrangeFESpace(mesh, p=2, doforder='sdofs')
