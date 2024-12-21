from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm

bm.set_backend('numpy')

from fealpy.functionspace import LagrangeFESpace

p = 3
n = 64
tmr = timer()
next(tmr)
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import QuadrangleMesh
pde = CosCosData()
mesh = QuadrangleMesh.from_box([0,1,0,1], n, n)

tmr.send('网格和pde生成时间')

space= LagrangeFESpace(mesh, p=p)
GD = mesh.geo_dimension()
ldof = space.number_of_local_dofs()
cm = mesh.entity_measure('cell')
q = p+1
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
NQ = ws.shape[0]
gphi = space.grad_basis(bcs, variable='x')  # (NC, NQ, LDOF, GD)
gphiu = space.grad_basis(bcs, variable='u')
M = bm.einsum('qim, qjn, q -> ijmnq', gphiu, gphiu, ws) # (ldof, ldof, GD, GD, NQ)
# M = bm.einsum('qim, qjn, q -> qijmn', gphiu, gphiu, ws) # (NQ, ldof, ldof, GD, GD)
J = mesh.jacobi_matrix(bcs)
G = mesh.first_fundamental_form(J)
G = bm.linalg.inv(G)
JG = bm.einsum("cqkm, cqmn -> cqkn", J, G)   # (NC, NQ, GD, GD)
tmr.send('准备')

result1_xx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[...,0], gphi[...,0], cm) 
result1_yy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[...,1], gphi[...,1], cm) 
result1_xy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[...,0], gphi[...,1], cm) 
result1_yx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[...,1], gphi[...,0], cm) 
tmr.send('普通: 先切片再 Einsum')

result1 = bm.einsum('q, cqim, cqjn, c -> cijmn', ws, gphi, gphi, cm) # (NC, ldof, ldof, GD, GD)
result1_xx1 = bm.copy(result1[...,0, 0])
result1_yy1 = bm.copy(result1[...,1, 1])
result1_xy1 = bm.copy(result1[...,0, 1])
result1_yx1 = bm.copy(result1[...,1, 0]) 
tmr.send('普通: 先 Einsum 再切片')


result2_xx = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,0,:], JG[...,0,:], cm) # (NC, NQ, ldof, GD)
result2_yy = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,1,:], JG[...,1,:], cm) # (NC, NQ, ldof, GD)
result2_xy = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,0,:], JG[...,1,:], cm) # (NC, NQ, ldof, GD)
result2_yx = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,1,:], JG[...,0,:], cm) # (NC, NQ, ldof, GD)

# result2_xy = np.einsum('qijmn, cqm, cqn, c -> cij', M, JG[...,0,:], JG[...,1,:], cm) # (NC, ldof, ldof)
# result2_xx = np.einsum('qijmn, cqm, cqn, c -> cij', M, JG[...,0,:], JG[...,0,:], cm) # (NC, ldof, ldof)
# result2_yy = np.einsum('qijmn, cqm, cqn, c -> cij', M, JG[...,1,:], JG[...,1,:], cm) # (NC, ldof, ldof)
# result2_yx = np.einsum('qijmn, cqm, cqn, c -> cij', M, JG[...,1,:], JG[...,0,:], cm) # (NC, ldof, ldof)
tmr.send('快速: 先切片再 Einsum')

# result2 = bm.einsum('cqam, qijmn, cqbn, c -> cijab', JG, M, JG, cm) # (NC, ldof, ldof, GD, GD)
result2 = bm.einsum('ijmnq, cqam, cqbn, c -> cijab', M, JG, JG, cm) # (NC, ldof, ldof, GD, GD)
result2_xx1 = bm.copy(result2[...,0, 0])
result2_yy1 = bm.copy(result2[...,1, 1])
result2_xy1 = bm.copy(result2[...,0, 1])
result2_yx1 = bm.copy(result2[...,1, 0])
tmr.send('快速: 先 Einsum 再切片')
next(tmr)

error = bm.sum(bm.abs(result1-result2))
error_xx = bm.sum(bm.abs(result1_xx-result2_xx))
error_yy = bm.sum(bm.abs(result1_yy-result2_yy))
error_xy = bm.sum(bm.abs(result1_xy-result2_xy))
error_yx = bm.sum(bm.abs(result1_yx-result2_yx))
print(f"error: {error}\n, error_xx: {error_xx}\n, error_yy: {error_yy}\n, error_xy: {error_xy}\n, error_yx: {error_yx}\n")
error_xx1 = bm.sum(bm.abs(result1_xx1-result1_xx))
error_yy1 = bm.sum(bm.abs(result1_yy1-result1_yy))
error_xy1 = bm.sum(bm.abs(result1_xy1-result1_xy))
error_yx1 = bm.sum(bm.abs(result1_yx1-result1_yx))
print(f"error_xx1: {error_xx1}\n, error_yy1: {error_yy1}\n, error_xy1: {error_xy1}\n, error_yx1: {error_yx1}\n")
print("-----------------")
    