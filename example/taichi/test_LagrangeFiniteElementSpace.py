
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF

from fealpy.decorator import cartesian
import taichi as ti
#from fealpy.ti import TriangleMesh 
#from fealpy.ti import TetrahedronMesh
#from fealpy.ti import LagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFiniteElementSpace as LFESpace

ti.init()

@cartesian
def usolution(p):
    x = p[...,0]
    y = p[...,1]
    u = np.zeros(p.shape)
    u[...,0] = 0.8*y*(1-y)
    return u
node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)

lmesh = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri')
#mesh = TriangleMesh(node, cell)

#space = LagrangeFiniteElementSpace(mesh, p=2)
lspace = LFESpace(lmesh,p=1)

u0 = lspace.interpolation(usolution,dim=2)

qf = lmesh.integrator(4)
bcs,ws = qf.get_quadrature_points_and_weights()
print(ws)
print(bcs)
#NN = mesh.number_of_nodes()
#NC = mesh.number_of_cells()
#NQ = len(ws)
#ldof = space.number_of_local_dofs()

print(bcs.shape)

'''
lgphi = lspace.grad_basis(bcs)
lphi = lspace.basis(bcs)

R0,R1 = space.shape_function(bcs)
phi = ti.field(ti.f64, shape=(NQ, ldof))
phi.from_numpy(R0)
gphi = ti.field(ti.f64, shape=R1.shape)
gphi.from_numpy(R1)
result1 = ti.field(ti.f64, shape=(NC, ldof, ldof))
result2 = ti.field(ti.f64, shape=(NC, ldof, ldof))
w = ti.field(ti.f64, shape=(NQ, ))
w.from_numpy(ws)
temp = ti.field(ti.f64,shape=(NC,ldof,ldof))

cellmeasure = lmesh.entity_measure('cell')


space.cell_pTgpy_matrices(result1,phi,gphi,w,None)
space.cell_gpyTp_matrices(result2,phi,gphi,w,None)
A = result1.to_numpy()
B = result2.to_numpy()
B = np.einsum('jmn->jnm',B)
print(A-B)
'''
## 测试 \nabla u^T \cdot u
'''
D1 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,u0(bcs)[...,0],lphi,lgphi[...,0],cellmeasure) 
#D2 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,u0(bcs)[...,1],lphi,lgphi[...,1],cellmeasure) 
D2 = np.einsum('i,ijk,ijm,j->jkm',ws,lphi,lgphi[...,1],cellmeasure) 
ux = u0(bcs)[...,0]
uy = u0(bcs)[...,1]
Cx = ti.field(ti.f64,shape=ux.shape)
Cy = ti.field(ti.f64,shape=uy.shape)
Cx.from_numpy(ux)
Cy.from_numpy(uy)
#space.cell_pTgpx_matrices(result,phi,gphi,Cx,w)
#print(result.to_numpy()-D1)
space.cell_pTgpy_matrices(result,phi,gphi,w,None)
#print(result.to_numpy()-D2)
'''

## 测试u^T \nabla \cdot u
'''
A1 = np.einsum('i,ijk,ijm,j->jkm',ws,lgphi[...,0],lphi,cellmeasure) 
A2 = np.einsum('i,ijk,ijm,j->jkm',ws,lgphi[...,1],lphi,cellmeasure) 
#space.cell_gpxTp_matrices(result,phi,gphi,w,None)
#print(result.to_numpy()-A1)
space.cell_gpyTp_matrices(result,phi,gphi,w,None)
print(result.to_numpy()-A2)
'''




## 测试单元刚度矩阵
'''
space.cell_stiff_matrices(result,gphi,w)
S = np.einsum('i,ijmk,ijnk,j->jmn',ws,lgphi,lgphi,lmesh.entity_measure('cell'))
print(result.to_numpy()-S)
'''
## 测试单元质量矩阵
'''
temp = ti.field(ti.f64, shape=(ldof,ldof))
result = ti.field(ti.f64, shape=(NC, ldof, ldof))
space.cell_mass_matrices(result, phi, w, temp)

M = np.einsum('i,ijm,ijn,j->jmn',ws,lphi,lphi,lmesh.entity_measure('cell'))
#print(M-result.to_numpy())
'''
