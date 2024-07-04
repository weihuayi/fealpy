import unittest
from unittest import TestCase
import torch 
from torch import Tensor
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator, BilinearForm, ScalarSourceIntegrator


class TestLinearElasticityAssembly(TestCase):
    def source(self, p: Tensor):
        """
        @brief 模型的源项值 f
        """

        x = p[..., 0]
        y = p[..., 1]
        val = Tensor.zeros(p.shape, dtype=Tensor.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

        return val

    def test_assembly(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        NX = 4
        NY = 4
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
        space = LagrangeFESpace(mesh, p=1, ctype='C')
        print("gdof:", space.number_of_global_dofs())
        tensor_space = TensorFunctionSpace(space, shape=(2, ), dof_last=True)

        integrator1 = LinearElasticityIntegrator(e=1.0, nu=0.3, elasticity_type='strain', device=device)
        KK = integrator1.assembly(space=tensor_space)
        
        bform = BilinearForm(tensor_space)
        K = bform.assembly()

        integrator2 = ScalarSourceIntegrator()


        print("K:", K)




        assert torch.allclose(KK_torch, torch.tensor(KK_expected), atol=1e-9)

if __name__ == '__main__':
    unittest.main(buffer=False)
