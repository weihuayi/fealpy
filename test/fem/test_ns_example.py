#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_ns_example.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年04月11日 星期二 18时42分28秒
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.fem import DiffusionIntegrator, MassIntegrator, ConvectionIntegrator
from fealpy.decorator import cartesian, barycentric
from fealpy.fem import LinearForm
from fealpy.fem import BilinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import SourceIntegrator

domain = [0, 1, 0, 1]
T=2
nt=100
ns = 20
p = 2
mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
space = LagrangeFESpace(mesh,p=p)
timeline = UniformTimeLine(0,T,nt)
dt = timeline.dt

bform = BilinearForm(2*(space,))
bform.add_domain_integrator(MassIntegrator())
bform.assembly()
A = bform.get_matrix()
