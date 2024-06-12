import torch
from torch import Tensor


CONTEXT = 'torch'

from fealpy.geometry import RectangleDomain
from fealpy.mesh import TriangleMesh as TMD

from fealpy.utils import timer

from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from fealpy.torch.solver import sparse_cg

import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PDEData:
    def __init__(self, cx=0.5, cy=0.5, eps=0.001):
        self.para = (cx, cy, eps)

    def solution(self, points: Tensor):
        cx, cy, eps = self.para
        x = points[..., 0]
        y = points[..., 1]
        val = (x - cx)**2 + (y - cy)**2 + eps 
        return 1.0/val

    def source(self, points: Tensor):
        cx, cy, eps = self.para
        x = points[..., 0]
        y = points[..., 1]
        val = (x - cx)**2 + (y - cy)**2
        return 4*(eps - val)/(eps + val)**3

    def dirichlet(self, points: Tensor):
        return self.solution(points) 

class PDEData_0:
    def solution(self, points: Tensor):
        pi = torch.pi
        x = points[..., 0]
        y = points[..., 1]
        return torch.sin(pi*x)*torch.sin(pi*y)

    def source(self, points: Tensor):
        pi = torch.pi
        x = points[..., 0]
        y = points[..., 1]
        return 2*pi**2*self.solution(points) 

    def dirichlet(self, points: Tensor):
        return self.solution(points) 


t = timer()
next(t)

pde = PDEData()
domain = RectangleDomain(hmin=0.1)
mesh0 = TMD.from_domain_distmesh(domain, maxit=1000)
node = mesh0.entity('node')
cell = mesh0.entity('cell')
t.send("np_mesh")

node = torch.tensor(node, device=device, requires_grad=True)
cell = torch.tensor(cell, device=device)
mesh = TriangleMesh(node, cell)
t.send("torch_mesh")

space = LagrangeFESpace(mesh, p=1)
gdof = space.number_of_global_dofs()
t.send('space')

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(method="fast_assembly"))

lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source))
t.send('forms')

A = bform.assembly()
F = lform.assembly()
t.send('assembly')


A, F = DirichletBC(space).apply(A, F, gd=pde.dirichlet)
t.send('dirichlet')

A = A.to_sparse_csr()
uh = sparse_cg(A, F, maxiter=1000)
t.send('solve')


cm = mesh.entity_measure('cell')
qf = mesh.integrator(3, etype='cell')
bcs, ws = qf.get_quadrature_points_and_weights()
ps = mesh.bc_to_point(bcs)
phi = space.basis(bcs)
cell2dof = space.dof.cell_to_dof() # (NC, 3)
error = pde.solution(ps) - torch.einsum('qj, cj->qc', phi, uh[cell2dof]) 

e = torch.sqrt(torch.einsum('q, qc, qc, c->', ws, error, error, cm))
e.backward()
t.send('backward')
t.send('stop')

grad = node.grad
grad[torch.isnan(grad)] = 0
grad[torch.isinf(grad)] = 0

print(grad)
print("e:", e)

fig, axes = plt.subplots()
mesh0.add_plot(axes)
node = node.detach().cpu().numpy()
grad = grad.detach().cpu().numpy()
axes.quiver(
        node[:, 0], 
        node[:, 1], 
        grad[:, 0],
        grad[:, 1])
plt.show()





