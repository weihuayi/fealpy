import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.storage import MeshBlock, EntitySector
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.schema.entity_schema import EntityContext


BACKENDS = ["numpy"]
BACKEND_IDS = ["backend-numpy"]


class TestPrismSchema:
    @staticmethod
    def make_ctx():
        node = bm.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ], dtype=bm.float64)

        cell = bm.array([
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],
        ], dtype=bm.int32)

        storage = MeshBlock(positions=node)
        storage.add_sector(EntitySector("prism", cell), root=True)
        TopologyBuilder.construct(storage)
        return EntityContext(block=storage, sector=storage.get_sector("prism"))

    @staticmethod
    def bcs():
        return (
            bm.array([[1.0 / 2.0, 1.0 / 4.0, 1.0 / 4.0]], dtype=bm.float64),
            bm.array([[1.0 / 3.0, 2.0 / 3.0]], dtype=bm.float64),
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_multi_index(self, backend):
        bm.set_backend(backend)
        schema = self.make_ctx().sector.schema

        assert schema.name == "prism"
        assert schema.top_dim == 3

        mi = bm.to_numpy(schema.multi_index((2, 2), tensorprod=False))
        expected = np.array([
            [2, 0, 0, 2, 0],
            [1, 1, 0, 2, 0],
            [1, 0, 1, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 1, 1, 2, 0],
            [0, 0, 2, 2, 0],
            [2, 0, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 2, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 2, 1, 1],
            [2, 0, 0, 0, 2],
            [1, 1, 0, 0, 2],
            [1, 0, 1, 0, 2],
            [0, 2, 0, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 0, 2, 0, 2],
        ], dtype=np.int32)
        np.testing.assert_array_equal(mi, expected)

        mi_tp = bm.to_numpy(schema.multi_index((2, 2)))
        assert mi_tp.shape == (18, 6)

        qf = schema.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        assert isinstance(bcs, tuple)
        assert len(bcs) == 2
        assert bcs[0].shape[-1] == 3
        assert bcs[1].shape[-1] == 2
        assert ws.shape[0] == bcs[0].shape[0] * bcs[1].shape[0]

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_shape_function(self, backend):
        bm.set_backend(backend)
        ctx = self.make_ctx()
        schema = ctx.sector.schema
        bcs = self.bcs()

        phi = schema.shape_function(bcs, p=(1, 1))
        grad_bary = schema.grad_shape_function_barycentric(bcs, p=(1, 1))
        grad_ref = schema.grad_shape_function_reference(bcs, p=(1, 1))
        grad_ref_legacy = schema.grad_shape_function(ctx, bcs, p=(1, 1), variables="u")

        # Contract order: bottom triangle vertices followed by top vertices.
        expected_phi = np.array([[
            1.0 / 6.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 3.0,
            1.0 / 6.0,
            1.0 / 6.0,
        ]])

        # Gradients use the same bottom-then-top vertex order.
        expected_grad_bary = np.array([[
            [1.0 / 3.0, 0.0,       0.0,       1.0 / 2.0, 0.0],
            [0.0,       1.0 / 3.0, 0.0,       1.0 / 4.0, 0.0],
            [0.0,       0.0,       1.0 / 3.0, 1.0 / 4.0, 0.0],
            [2.0 / 3.0, 0.0,       0.0,       0.0,       1.0 / 2.0],
            [0.0,       2.0 / 3.0, 0.0,       0.0,       1.0 / 4.0],
            [0.0,       0.0,       2.0 / 3.0, 0.0,       1.0 / 4.0],
        ]])

        expected_grad_ref = np.array([[
            [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 2.0],
            [1.0 / 3.0, 0.0, -1.0 / 4.0],
            [0.0, 1.0 / 3.0, -1.0 / 4.0],
            [-2.0 / 3.0, -2.0 / 3.0, 1.0 / 2.0],
            [2.0 / 3.0, 0.0, 1.0 / 4.0],
            [0.0, 2.0 / 3.0, 1.0 / 4.0],
        ]])

        assert phi.shape == (1, 6)
        assert grad_bary.shape == (1, 6, 5)
        assert grad_ref.shape == (1, 6, 3)
        np.testing.assert_allclose(bm.to_numpy(phi), expected_phi, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(grad_bary), expected_grad_bary, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(grad_ref), expected_grad_ref, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(grad_ref_legacy), expected_grad_ref, atol=1.0e-12)

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_geometry(self, backend):
        bm.set_backend(backend)
        ctx = self.make_ctx()
        schema = ctx.sector.schema

        point = schema.bc_to_point(ctx, self.bcs(), None)
        point_index = schema.bc_to_point(ctx, self.bcs(), slice(1, 2))
        barycenter = schema.barycenter(ctx, None)
        barycenter_index = schema.barycenter(ctx, slice(1, 2))
        measure = schema.measure(ctx, None)
        normal = schema.normal(ctx, None)
        tangent = schema.tangent(ctx, None)
        tangent_index = schema.tangent(ctx, slice(1, 2))

        # The prism contract is bottom vertices followed by top vertices.
        expected_point = np.array([
            [[1.0 / 4.0, 1.0 / 4.0,  2.0 / 3.0]],
            [[1.0 / 4.0, 1.0 / 4.0,  5.0 / 3.0]],
        ])

        expected_barycenter = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 1.5],
        ])

        expected_measure = np.array([0.5, 0.5])
        expected_tangent = np.tile(np.eye(3), (2, 1, 1))

        np.testing.assert_allclose(bm.to_numpy(point), expected_point, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(point_index), expected_point[1:2], atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(barycenter), expected_barycenter, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(barycenter_index), expected_barycenter[1:2], atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(measure), expected_measure, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(tangent), expected_tangent, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(tangent_index), expected_tangent[1:2], atol=1.0e-12)
        assert normal.shape == (2, 0, 3)
        assert schema.geo_dimension(ctx) == 3

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_jacobi(self, backend):
        bm.set_backend(backend)
        ctx = self.make_ctx()
        schema = ctx.sector.schema
        bcs = self.bcs()

        J = schema.jacobi_matrix(ctx, bcs, None)
        J_index = schema.jacobi_matrix(ctx, bcs, slice(1, 2))
        G = schema.first_fundamental_form(ctx, bcs, index=None)
        G_index = schema.first_fundamental_form(ctx, bcs, index=slice(1, 2))
        G2, J2 = schema.first_fundamental_form(ctx, bcs, index=None, return_jacobi=True)
        G3, gphi = schema.first_fundamental_form(ctx, bcs, index=None, return_grad=True)

        expected_J = np.tile(np.eye(3), (2, 1, 1, 1))
        expected_G = np.tile(np.eye(3), (2, 1, 1, 1))
        expected_grad_ref = np.array([[
            [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 2.0],
            [1.0 / 3.0, 0.0, -1.0 / 4.0],
            [0.0, 1.0 / 3.0, -1.0 / 4.0],
            [-2.0 / 3.0, -2.0 / 3.0, 1.0 / 2.0],
            [2.0 / 3.0, 0.0, 1.0 / 4.0],
            [0.0, 2.0 / 3.0, 1.0 / 4.0],
        ]])

        assert J.shape == (2, 1, 3, 3)
        assert G.shape == (2, 1, 3, 3)
        np.testing.assert_allclose(bm.to_numpy(J), expected_J, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(J_index), expected_J[1:2], atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(G), expected_G, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(G_index), expected_G[1:2], atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(G2), expected_G, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(J2), expected_J, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(G3), expected_G, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(gphi), expected_grad_ref, atol=1.0e-12)


if __name__ == "__main__":
    # pytest.main(["./test_prism_schema.py", "-k", "test_multi_index"])
    # pytest.main(["./test_prism_schema.py", "-k", "test_shape_function"])
    # pytest.main(["./test_prism_schema.py", "-k", "test_geometry"])
    # pytest.main(["./test_prism_schema.py", "-k", "test_jacobi"])
    pytest.main(["./test_prism_schema.py"])