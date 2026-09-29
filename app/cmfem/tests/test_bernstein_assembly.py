import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import BernsteinFESpace
from fealpy.mesh import TetrahedronMesh, TriangleMesh

from bernstein_assembly import bernstein_stiffness
from cm_fe_space2d import CmFESpace2d
from cm_fe_space3d import CmFESpace3d
from symmetric_tensor import symmetry_multiplicity


def assert_allclose(value, expected, rtol=1.0e-7, atol=0.0):
    assert bool(bm.allclose(value, expected, rtol=rtol, atol=atol))


class TestBernsteinAssembly(unittest.TestCase):
    def setUp(self):
        bm.set_backend('numpy')

    def test_lambda_derivative(self):
        for dim in (2, 3):
            mesh = (TriangleMesh.from_box(nx=1, ny=1) if dim == 2 else
                    TetrahedronMesh.from_box(nx=1, ny=1, nz=1))
            bcs = (bm.array([[0.2, 0.3, 0.5]]) if dim == 2 else
                   bm.array([[0.1, 0.2, 0.3, 0.4]]))
            for p, order in ((3, 1), (4, 2), (5, 3)):
                with self.subTest(dim=dim, p=p, order=order):
                    space = BernsteinFESpace(mesh, p=p, ctype='D')
                    derivative, transform = space.grad_m_basis(
                        bcs, order, variable='lambda'
                    )
                    value = bm.einsum(
                        'aqi,acg->cqig', derivative, transform
                    )
                    expected = space.grad_m_basis(bcs, order)
                    assert_allclose(value, expected, atol=1.0e-12)

    def test_lambda_integral_has_no_cell_axis(self):
        for dim in (2, 3):
            mesh = (TriangleMesh.from_box(nx=2, ny=2) if dim == 2 else
                    TetrahedronMesh.from_box(nx=2, ny=2, nz=2))
            space = BernsteinFESpace(mesh, p=4, ctype='D')
            bcs, weights = mesh.quadrature_formula(
                5, 'cell'
            ).get_quadrature_points_and_weights()
            derivative, _ = space.grad_m_basis(
                bcs, 2, variable='lambda'
            )
            value = bm.einsum(
                'aqi,bqj,q->abij', derivative, derivative, weights
            )
            nlambda = len(bm.multi_index_matrix(2, dim, dtype=bm.int32))
            ldof = len(bm.multi_index_matrix(4, dim, dtype=bm.int32))
            self.assertEqual(value.shape, (nlambda, nlambda, ldof, ldof))

    def test_bernstein_stiffness(self):
        meshes = [
            TriangleMesh(
                bm.array([[0.1, 0.2], [2.0, 0.3], [0.4, 1.7]]),
                bm.array([[0, 1, 2]], dtype=bm.int32)
            ),
            TetrahedronMesh(
                bm.array([[0.1, 0.2, 0.1], [2.0, 0.3, 0.2],
                          [0.4, 1.7, 0.3], [0.2, 0.5, 1.9]]),
                bm.array([[0, 1, 2, 3]], dtype=bm.int32)
            ),
        ]
        for mesh in meshes:
            dim = mesh.top_dimension()
            for order in (1, 2, 3):
                p = order+3
                with self.subTest(dim=dim, order=order):
                    bcs, weights = mesh.quadrature_formula(
                        p+1, 'cell'
                    ).get_quadrature_points_and_weights()
                    space = BernsteinFESpace(mesh, p=p, ctype='D')
                    derivative = space.grad_m_basis(bcs, order)
                    multiplicity = symmetry_multiplicity(
                        order, dim, dtype=mesh.ftype
                    )
                    expected = bm.einsum(
                        'cqig,cqjg,g,q,c->cij', derivative, derivative,
                        multiplicity, weights, mesh.entity_measure('cell')
                    )

                    local_space = SimpleNamespace(
                        p=p, mesh=mesh, bspace=space,
                        lattice=SimpleNamespace(permutation=bm.arange(
                            expected.shape[-1], dtype=bm.int32
                        ))
                    )
                    value = bernstein_stiffness(local_space, order, p+1)
                    assert_allclose(value, expected, atol=1.0e-10)

    def test_smooth_stiffness(self):
        cases = [
            (CmFESpace2d, TriangleMesh.from_box(nx=1, ny=1),
             5, (2, 1, 0), (2, 3)),
            (CmFESpace2d, TriangleMesh.from_box(nx=1, ny=1),
             9, (4, 2, 0), (3,)),
            (CmFESpace3d, TetrahedronMesh.from_box(nx=1, ny=1, nz=1),
             9, (4, 2, 1, 0), (2, 3)),
        ]
        for space_type, mesh, p, r, orders in cases:
            space = space_type(mesh, p, r)
            for order in orders:
                with self.subTest(dim=mesh.top_dimension(), p=p, order=order):
                    expected = space.stiffness_matrix(
                        order=order, q=p+1, method='quadrature'
                    ).to_scipy()
                    value = space.stiffness_matrix(
                        order=order, q=p+1
                    ).to_scipy()
                    relative = bm.linalg.norm(
                        (value-expected).data
                    )/bm.linalg.norm(expected.data)
                    symmetry = bm.linalg.norm(
                        (value-value.T).data
                    )/bm.linalg.norm(value.data)
                    self.assertLess(relative, 1.0e-11)
                    self.assertLess(symmetry, 1.0e-13)

    def test_stiffness_uses_coefficient_matrix(self):
        cases = [
            CmFESpace2d(TriangleMesh.from_box(nx=1, ny=1),
                        5, (2, 1, 0)),
            CmFESpace3d(TetrahedronMesh.from_box(nx=1, ny=1, nz=1),
                        9, (4, 2, 1, 0)),
        ]
        for space in cases:
            self.assertIsNone(space._coeff)
            space.stiffness_matrix()
            self.assertIsNotNone(space._coeff)


if __name__ == '__main__':
    unittest.main()
