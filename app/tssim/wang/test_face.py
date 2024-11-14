#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_face.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 05 Nov 2024 03:54:38 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import TensorFunctionSpace, LagrangeFESpace
from tangent_face_mass_integrator import TangentFaceMassIntegrator
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)
from fealpy.decorator import barycentric,cartesian
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.sparse import COOTensor, CSRTensor

q=1
@cartesian
def is_wall_boundary(p):
    return (bm.abs(p[..., 1]) < 1e-10) | \
           (bm.abs(p[..., 1] - 1) < 1e-10)

mesh = TriangleMesh.from_box([0, 1, 0, 1], 4, 4)
space = LagrangeFESpace(mesh, p=1)
tspace = TensorFunctionSpace(space, shape=(2,-1))

qf = mesh.quadrature_formula(q, 'face')
bcs, ws = qf.get_quadrature_points_and_weights()
print(mesh.bc_to_point(bcs).shape)

exit()

'''
index = mesh.boundary_face_index()
bc = mesh.entity_barycenter('face', index=index)
index = index[pde.is_wall_boundary(bc)]
print(index)
#print(mesh.edge_unit_normal(index))
print(mesh.edge_unit_tangent(index))
ipoint = space.interpolation_points()
import matplotlib.pylab  as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_edge(axes,fontsize=20,showindex=True)
plt.show()
exit()
'''

Ab = BilinearForm(tspace)
Ab.add_integrator(TangentFaceMassIntegrator(coef=1, q=q, threshold=is_wall_boundary))
A = Ab.assembly()



index = mesh.boundary_face_index()
bc = mesh.entity_barycenter('face', index=index)
index = index[is_wall_boundary(bc)]

tangent = mesh.edge_unit_tangent()[index]
qf = mesh.quadrature_formula(q, 'face')
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space.face_basis(bcs)
fm = mesh.entity_measure('face', index=index)

t0phi = bm.einsum('e, eqi->eqi', tangent[:, 0], phi)
t1phi = bm.einsum('e, eqi->eqi', tangent[:, 1], phi)
t0_cell = bm.einsum('e, q, eqi, eqj -> eij', fm, ws, phi, t0phi)
t1_cell = bm.einsum('e, q, eqi, eqj -> eij', fm, ws, phi, t1phi)
face_to_dof = space.face_to_dof(index)
gdof = space.number_of_global_dofs()

M = COOTensor(
    indices = bm.empty((2, 0), dtype=space.itype, device=bm.get_device(space)),
    values = bm.empty((0,), dtype=space.ftype, device=bm.get_device(space)),
    spshape = (gdof, gdof)
)
I = bm.broadcast_to(face_to_dof[:, :, None], t0_cell.shape)
J = bm.broadcast_to(face_to_dof[:, None, :], t0_cell.shape)
indices = bm.stack([I.ravel(), J.ravel()], axis=0)
M0 = M.add(COOTensor(indices, t0_cell.reshape(-1), (gdof, gdof)))

M = COOTensor(
    indices = bm.empty((2, 0), dtype=space.itype, device=bm.get_device(space)),
    values = bm.empty((0,), dtype=space.ftype, device=bm.get_device(space)),
    spshape = (gdof, gdof)
)
I = bm.broadcast_to(face_to_dof[:, :, None], t0_cell.shape)
J = bm.broadcast_to(face_to_dof[:, None, :], t0_cell.shape)
indices = bm.stack([I.ravel(), J.ravel()], axis=0)
M1 = M.add(COOTensor(indices, t1_cell.reshape(-1), (gdof, gdof)))
AA0 = COOTensor.concat([M0, M1], axis=1)
AA1 = COOTensor.concat([M0, M1], axis=1)
AA = COOTensor.concat([AA0, AA1], axis=0)

'''
AAA = BilinearForm(space)
AAA.add_integrator(BoundaryFaceMassIntegrator(coef=tangent[:,1], q=q, threshold=is_wall_boundary))
AAA = AAA.assembly()
'''

#print(AA.to_dense()[:gdof,gdof:])
#print(A.to_dense()[:gdof,:gdof])
#print(AA.to_dense()[:gdof,gdof:] -A.to_dense()[:gdof,gdof:])
print(bm.sum(bm.abs(AA.to_dense()-A.to_dense())))
'''
import matplotlib.pylab  as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_edge(axes,fontsize=20,showindex=True)
plt.show()
'''
'''
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.fem import (ScalarConvectionIntegrator, 
                        ScalarDiffusionIntegrator, 
                        ScalarMassIntegrator,
                        SourceIntegrator,
                        PressWorkIntegrator)
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)
from tangent_face_mass_integrator import TangentFaceMassIntegrator
a1 = BoundaryFaceMassIntegrator(coef=1, q=4, threshold=pde.is_up_boundary)     
a2 = BoundaryFaceMassIntegrator(coef=-1, q=4, threshold=pde.is_down_boundary)     
a3 = TangentFaceMassIntegrator(coef=1, q=4, threshold=pde.is_wall_boundary)
AA = BilinearForm(space)
AA.add_integrator(a1)
AA.add_integrator(a2)
AA = AA.assembly()
BB = BilinearForm(uspace)
BB.add_integrator(a3)
BB = BB.assembly()
ndof = space.number_of_global_dofs()
print(bm.sum(bm.abs(AA.to_dense()+BB.to_dense()[:ndof,:ndof])))
#print(bm.sum(bm.abs(AA.to_dense())))
#print(bm.sum(bm.abs(BB.to_dense()[ndof:,:ndof])))
exit()
'''
