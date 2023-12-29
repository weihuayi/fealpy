#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_matrix.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年05月05日 星期五 11时34分55秒
	@bref 
	@ref 
'''  
import numpy as np
import pytest
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.fem import VectorMassIntegrator,VectorDiffusionIntegrator 
from fealpy.fem import VectorConvectionIntegrator 
from fealpy.fem import ScalarConvectionIntegrator 
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace 
from scipy.sparse import bmat
from fealpy.decorator import cartesian

from tssim.part import *
from tssim.part.Assemble import Assemble

T=2
nt=50
ns = 16
udegree = 2
pdegree = 1
eps = 1e-9

mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
uspace = LagrangeFESpace(mesh,p=udegree,doforder='sdofs')
pspace = LagrangeFESpace(mesh,p=pdegree,doforder='sdofs')

Ouspace = LagrangeFiniteElementSpace(mesh,p=udegree)
Opspace = LagrangeFiniteElementSpace(mesh,p=pdegree)
assemble =  Assemble(mesh,4)

def test_vector_mass():
    Vbform = BilinearForm((uspace,)*2)
    Vbform.add_domain_integrator(VectorMassIntegrator(c=1))
    Vbform.assembly()
    M = Vbform.get_matrix()
    OM = assemble.matrix([udegree, 0],[udegree, 0])
    OM = bmat([[OM,None],[None,OM]])
    assert np.sum(np.abs(OM.toarray()-M.toarray())) < eps

def test_scalar_diffusion():
    Vbform = BilinearForm((uspace,)*2)
    Vbform.add_domain_integrator(VectorDiffusionIntegrator(c=1))
    Vbform.assembly()
    S = Vbform.get_matrix()
    S1 = assemble.matrix([udegree, 1], [udegree, 1])
    S2 = assemble.matrix([udegree, 2], [udegree, 2])
    OS = S1+S2
    OS = bmat([[OS,None],[None,OS]])
    assert np.sum(np.abs(S.toarray()-OS.toarray())) < eps

'''
def test_vector_convection(): 
    @cartesian
    def velocity_field(p):
        x = p[..., 0]
        y = p[..., 1]
        u = np.zeros(p.shape)
        u[..., 0] = np.sin(np.pi * x) ** 2 * np.sin(2 * np.pi * y)
        u[..., 1] = -np.sin(np.pi * y) ** 2 * np.sin(2 * np.pi * x)
        return u
    u0=uspace.interpolate(velocity_field)
    Ou0=Ouspace.interpolation(velocity_field)
 
    Vbform = BilinearForm((uspace,)*2)
    Vbform.add_domain_integrator(VectorConvectionIntegrator(c=u0))
    Vbform.assembly()
    C = Vbform.get_matrix()

    Sbform = BilinearForm(uspace)
    Sbform.add_domain_integrator(ScalarConvectionIntegrator(c=u0))
    Sbform.assembly()
    SC = Sbform.get_matrix()
    
    C1 = assemble.matrix([udegree, 1], [udegree, 0], Ou0(assemble.bcs)[...,0])
    C2 = assemble.matrix([udegree, 2], [udegree, 0], Ou0(assemble.bcs)[...,1])
    OC = C1+C2
    #OC = bmat([[OC,None],[None,OC]])
    print(np.sum(np.abs(SC.toarray()-OC.toarray())))
    assert np.sum(np.abs(SC.toarray()-OC.toarray())) < eps
'''
