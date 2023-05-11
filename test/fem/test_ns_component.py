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
from fealpy.fem import VectorMassIntegrator,ScalarDiffusionIntegrator 
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace 
from scipy.sparse import bmat

T=2
nt=50
ns = 16
p = 2
mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
space = LagrangeFESpace(mesh,p=p,doforder='sdofs')
Vbform = BilinearForm((space,space))
Sbform = BilinearForm(space)
oldspace = LagrangeFiniteElementSpace(mesh,p=p)

def test_vector_mass():
    Vbform.add_domain_integrator(VectorMassIntegrator(c=1,q=3))
    Vbform.assembly()
    VM = Vbform.get_matrix()
    VM0 = oldspace.mass_matrix(c=1,q=3)
    VM0 = bmat([[VM0,None],[None,VM0]])
    print(np.sum(np.abs(VM0.toarray()-VM.toarray())))
    #assert np.allclose(VM0.toarray(), VM.toarray(), rtol=0.0001, atol=0.0005)

def test_scalar_diffusion():
    Sbform.add_domain_integrator(ScalarDiffusionIntegrator(c=1,q=3))
    Sbform.assembly()
    SD = Sbform.get_matrix()
    SD0 = oldspace.stiff_matrix()
    print(np.sum(np.abs(SD.toarray()-SD0.toarray())))

''''
def test_vector_diffusion():
    VMI = VectorMassIntegrator(space)
    bform.add_domain_integrator(VectorMassIntegrator(c=1,q=3))
    bform.assembly()
    VM = bform.get_matrix()
    VM0 = oldspace.mass_matrix()
    VM0 = bmat([[VM0,None],[None,VM0]])
    #print(np.sum(np.abs(VM0.toarray()-VM.toarray())))
    assert VM0.shape == VM.shape
    assert VM0.nnz == VM.nnz
    assert (VM0 != VM).nnz == 0
    assert np.allclose(VM0.toarray(), VM.toarray())
'''
test_vector_mass()
test_scalar_diffusion()
