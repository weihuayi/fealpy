import numpy as np
import pytest
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem.linear_elasticity_integrator import LinearElasticityIntegrator
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.fem.linear_form import LinearForm
from fealpy.experimental.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.sparse.linalg import sparse_cg
from fealpy.experimental.mesh import UniformMesh2d, TriangleMesh
from fealpy.experimental.tests.fem.linear_elasticity_integrator_data import *

class TestLinearElasticityIntegratorInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def triangle_strain_p1(self, meshdata, backend):
        bm.set_backend(backend)

        def source(points: TensorLike) -> TensorLike:
            x = points[..., 0]
            y = points[..., 1]
            
            val = bm.zeros(points.shape, dtype=points.dtype)
            val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
            val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
            
            return val

        def solution(points: TensorLike) -> TensorLike:
            x = points[..., 0]
            y = points[..., 1]
            
            val = bm.zeros(points.shape, dtype=points.dtype)
            val[..., 0] = x * (1 - x) * y * (1 - y)
            val[..., 1] = 0
            
            return val

        def dirichlet(points: TensorLike) -> TensorLike:

            return solution(points)

        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2)
        maxit = 5
        errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
        for i in range(maxit):
            p = 2
            space = LagrangeFESpace(mesh, p=p, ctype='C')
            tensor_space = TensorFunctionSpace(space, shape=(2, -1))
            # (tgdof, )
            uh = tensor_space.function()
            # 与单元有关的组装方法
            integrator_bi_dependent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                                            elasticity_type='strain', q=5)
            # 与单元无关的组装方法
            integrator_bi_independent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                                            method='fast_strain', q=5)
            
            # 与单元有关的组装方法
            KK_dependent = integrator_bi_dependent.assembly(space=tensor_space)
            # 与单元无关的组装方法
            KK_independent = integrator_bi_independent.fast_assembly_strain_constant(space=tensor_space)
            
            # 与单元有关的组装方法
            bform_dependent = BilinearForm(tensor_space)
            bform_dependent.add_integrator(integrator_bi_dependent)
            K_dependent = bform_dependent.assembly()
            # 与单元无关的组装方法
            bform_independent = BilinearForm(tensor_space)
            bform_independent.add_integrator(integrator_bi_independent)
            K_independent = bform_independent.assembly()

            integrator_li = VectorSourceIntegrator(source=source, q=5)
            FF = integrator_li.assembly(space=tensor_space)

            lform = LinearForm(tensor_space)
            lform.add_integrator(integrator_li)
            F = lform.assembly()

            dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
            isDDof = tensor_space.is_boundary_dof(threshold=None)
            K = dbc.check_matrix(K)
            kwargs = K.values_context()
            indices = K.indices()
            new_values = bm.copy(K.values())
            IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
            new_values[IDX] = 0
            K = COOTensor(indices, new_values, K.sparse_shape)
            index, = bm.nonzero(isDDof, as_tuple=True)
            one_values = bm.ones(len(index), **kwargs)
            one_indices = bm.stack([index, index], axis=0)
            K1 = COOTensor(one_indices, one_values, K.sparse_shape)
            K = K.add(K1).coalesce()

            F = dbc.check_vector(F)
            uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh)
            F = F - K.matmul(uh[:])
            F[isDDof] = uh[isDDof]

            uh[:] = sparse_cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)

            u_exact = tensor_space.interpolate(solution)
            errorMatrix[0, i] = bm.max(bm.abs(bm.array(uh) - u_exact))
            errorMatrix[1, i] = bm.max(bm.abs(bm.array(uh) - u_exact)[isDDof])

            tgdof = tensor_space.number_of_global_dofs()
            assert K.shape == (tgdof, tgdof), \
            "Global stiffness matrix shape error!"
