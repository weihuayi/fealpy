#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_ns_fem_solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年12月06日 星期三 10时59分28秒
	@bref 
	@ref 
'''  
import pytest
import numpy as np
from scipy.sparse import spdiags, bmat, csr_matrix, hstack, vstack

from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace
#from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian
from fealpy.cfd import NSFEMSolver 


from tssim.part import *
from tssim.part.Assemble import Assemble
from tssim.part.Level_Set import Level_Set

@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin(np.pi * x) ** 2 * np.sin(2 * np.pi * y)
    u[..., 1] = -np.sin(np.pi * y) ** 2 * np.sin(2 * np.pi * x)
    return u

@pytest.fixture
def ns_solver_setup():
    q = 4
    mesh = TriangleMesh.from_box([0, 10, 0, 1], nx = 10, ny = 100) 
    assemble0 =  Assemble(mesh,q)
    
    mesh = TriangleMesh.from_box([0, 10, 0, 1], nx = 10, ny = 100)
    Ouspace = LagrangeFiniteElementSpace(mesh, p=2)
    q = 4
    
    Vfield = Ouspace.interpolation(velocity_field, dim=2)

    uspace = LagrangeFESpace(mesh, p=2, doforder='sdofs')
    pspace = LagrangeFESpace(mesh, p=1, doforder='sdofs')
    solver = NSFEMSolver(mesh,0.1,uspace,pspace,1,0.1)
    return assemble0, Vfield, solver

def test_ossen_A(ns_solver_setup):
    assemble, Vfield, solver= ns_solver_setup
    mu = 0.1
    dt = 0.1
    
    ## 老接口
    udegree = 2
    pdegree = 1
    M = assemble.matrix([udegree, 0],[udegree, 0])

    C1 = assemble.matrix([pdegree, 0],[udegree, 1])
    C2 = assemble.matrix([pdegree, 0],[udegree, 2])

    S1 = assemble.matrix([udegree, 1], [udegree, 1], mu)
    S2 = assemble.matrix([udegree, 2], [udegree, 2], mu)
    S = S1+S2
    
    D1 = assemble.matrix([udegree, 1], [udegree, 0], Vfield(assemble.bcs)[...,0])
    D2 = assemble.matrix([udegree, 2], [udegree, 0], Vfield(assemble.bcs)[...,1])
    D = D1+D2
    A = bmat([[1/dt*M+S+D, None,       -C1],\
            [None,         1/dt*M+S+D, -C2],\
            [C1.T,         C2.T,       None]],format = 'csr')
    
    ## 新接口
    AA = solver.ossen_A(Vfield)

    assert np.abs(np.sum(np.abs(A-AA))) < 1e-10

def test_ossen_b(ns_solver_setup):
    assemble, Vfield, solver= ns_solver_setup

    ## 老接口
    udegree = 2
    pdegree = 1
    dt = 0.1
    pgdof = assemble.mesh.number_of_global_ipoints(p=pdegree)
    M = assemble.matrix([udegree, 0],[udegree, 0])
    b = np.hstack((M@Vfield[..., 0], M@Vfield[..., 1]))    
    b = np.hstack((b, [0]*pgdof))
    b *= 1/dt
    
    ## 新接口
    uspace = LagrangeFESpace(assemble.mesh, p=2, doforder='sdofs')
    u0 = uspace.interpolate(velocity_field, dim=2)
    bb = solver.Ossen_b(u0)

    assert np.sum(np.abs(b-bb))< 1e-10
