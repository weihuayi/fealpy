from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from fealpy.experimental.mesh import TriangleMesh
from fealpy.geometry import RectangleDomain
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from .. import logger


class MovingMeshAlg:
    def __init__(self, mesh , uh , pde):
        self.mesh = mesh
        self.uh = uh
        self.pde = pde


    def __call__(self, node: TensorLike):
        node.requires_grad_(True) 
        E = self.get_energe(node)
        node.retain_grad()
        E.backward(retain_graph=True)
        grad = node.grad
        return E , grad
    
    def get_energy(self,node : TensorLike):
        p =1
        cell = self.mesh.cell
        isbdnode = self.mesh.boundary_node_flag()
        source = self.pde.source

        mesh = TriangleMesh(node,cell)     
        space = LagrangeFESpace(mesh,p)

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(method="fast"))
        A = bform.assembly()
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(source))
        F = lform.assembly()
        # 去除边界项的影响
        A = A.to_dense()
        A = A[~isbdnode]
        A = A[:,~isbdnode]
        F = F[~isbdnode]

        E = 1/2 * bm.einsum('i , ij ,j->', self.uh, A, self.uh) - bm.dot(self.uh,F)

