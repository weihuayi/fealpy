#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_face1.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Fri 08 Nov 2024 09:32:14 AM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)
from tangent_face_mass_integrator import TangentFaceMassIntegrator
from pde import CouetteFlow
from fealpy.decorator import barycentric,cartesian

pde = CouetteFlow()
h = 1/16
mesh = pde.mesh(h)
space = LagrangeFESpace(mesh, p=1)
uspace = TensorFunctionSpace(space, shape=(2,-1))

@cartesian
def coef_u0(p):
    x = p[..., 0]
    y = p[..., 1]
    val = bm.zeros(p.shape)
    val[..., 0] = x+y
    val[..., 1] = 2*x+2*y
    return val
u0 = uspace.interpolate(coef_u0)
qf = mesh.quadrature_formula(q=2, etype='cell')
bcs,ws = qf.get_quadrature_points_and_weights()
print(u0(bcs).shape)
print(u0.grad_value(bcs))
exit()
Bform = BilinearForm(uspace) 
FM = TangentFaceMassIntegrator(coef=1, q=5, threshold=pde.is_wall_boundary)
Bform.add_integrator(FM)
A = Bform.assembly().to_dense()

t = mesh.edge_unit_tangent()

@barycentric
def source_coef(bcs, index):
    result = bm.einsum('ijd, id->ij', u0(bcs, index), t[index,:])
    result = bm.repeat(result[..., bm.newaxis], 2, axis=-1)
    return result
Lform = LinearForm(uspace)
FS = BoundaryFaceSourceIntegrator(source=source_coef, q=5, threshold=pde.is_wall_boundary)
Lform.add_integrator(FS)
b  = Lform.assembly()

print(bm.sum(bm.abs(A@u0 - b)))

