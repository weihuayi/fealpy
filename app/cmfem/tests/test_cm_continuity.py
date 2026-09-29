import sys
import unittest
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cm_fe_space2d import CmFESpace2d


def assert_allclose(value, expected, rtol=1.0e-7, atol=0.0):
    assert bool(bm.allclose(value, expected, rtol=rtol, atol=atol))


class TestCmContinuity(unittest.TestCase):
    def setUp(self):
        bm.set_backend('numpy')
        node = bm.array([
            [0.0, 0.0], [1.2, 0.1], [1.0, 1.2],
            [-0.1, 0.9], [0.45, 0.55]
        ])
        cell = bm.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
        ], dtype=bm.int32)
        self.mesh = TriangleMesh(node, cell)

    def edge_barycentric_coordinates(self, edge, cell, t):
        """Return barycentric coordinates of edge points in one adjacent cell."""
        local = {int(vertex): i for i, vertex in enumerate(cell)}
        bcs = bm.zeros((len(t), 3))
        bcs[:, local[int(edge[0])]] = 1-t
        bcs[:, local[int(edge[1])]] = t
        return bcs

    def derivative_value(self, space, local_dof, cell, bcs, order):
        """Evaluate an order-th derivative on one cell with ``CmFESpace2d``."""
        phi = space.grad_m_basis(
            bcs, order, index=slice(cell, cell+1)
        )[0]
        if order == 0:
            return bm.einsum('qi,i->q', phi, local_dof)
        return bm.einsum('qig,i->qg', phi, local_dof)

    def vertex_derivatives(self, space, local_dof, cell, vertex, order):
        bcs = bm.zeros((1, 3))
        local_vertex = bm.where(self.mesh.entity('cell')[cell] == vertex)[0][0]
        bcs[0, local_vertex] = 1
        return self.derivative_value(space, local_dof, cell, bcs, order)

    def test_continuity_across_internal_edges(self):
        parameters = [
            (5, (2, 1, 0), 1),
            (7, (3, 1, 0), 1),
            (9, (4, 2, 0), 2),
            (11, (5, 2, 0), 2),
        ]
        t = bm.array([0.07, 0.23, 0.51, 0.82, 0.96])
        edge = self.mesh.entity('edge')
        cell = self.mesh.entity('cell')
        face_to_cell = self.mesh.face_to_cell()

        for p, r, m in parameters:
            with self.subTest(p=p, r=r):
                space = CmFESpace2d(self.mesh, p, r)
                bm.random.seed(2026)
                uh = bm.random.normal(size=space.number_of_global_dofs())
                local_dof = uh[space.cell_to_dof()]
                higher_order_jump = 0.0

                for e, (left, right, _, _) in enumerate(face_to_cell):
                    if left == right:
                        continue
                    left_bcs = self.edge_barycentric_coordinates(edge[e], cell[left], t)
                    right_bcs = self.edge_barycentric_coordinates(edge[e], cell[right], t)

                    for order in range(m+1):
                        left_value = self.derivative_value(
                            space, local_dof[left], left, left_bcs, order
                        )
                        right_value = self.derivative_value(
                            space, local_dof[right], right, right_bcs, order
                        )
                        assert_allclose(
                            left_value, right_value, rtol=2e-12, atol=2e-11
                        )

                    left_value = self.derivative_value(
                        space, local_dof[left], left, left_bcs, m+1
                    )
                    right_value = self.derivative_value(
                        space, local_dof[right], right, right_bcs, m+1
                    )
                    higher_order_jump = max(
                        higher_order_jump, bm.max(bm.abs(left_value-right_value))
                    )

                self.assertGreater(higher_order_jump, 1e-6)

    def test_smoothness_at_shared_vertices(self):
        parameters = [
            (5, (2, 1, 0)),
            (7, (3, 1, 0)),
            (9, (4, 2, 0)),
            (11, (5, 2, 0)),
        ]
        cell = self.mesh.entity('cell')

        for p, r in parameters:
            with self.subTest(p=p, r=r):
                space = CmFESpace2d(self.mesh, p, r)
                bm.random.seed(2026)
                uh = bm.random.normal(size=space.number_of_global_dofs())
                local_dof = uh[space.cell_to_dof()]
                higher_order_jump = 0.0

                for vertex in range(self.mesh.number_of_nodes()):
                    cells = bm.where(bm.any(cell == vertex, axis=1))[0]
                    if len(cells) < 2:
                        continue

                    for order in range(r[0]+1):
                        reference = self.vertex_derivatives(
                            space, local_dof[cells[0]], cells[0], vertex, order
                        )
                        for c in cells[1:]:
                            value = self.vertex_derivatives(
                                space, local_dof[c], c, vertex, order
                            )
                            relative_error = bm.max(
                                bm.abs(value-reference)/(1+bm.abs(value)+bm.abs(reference))
                            )
                            self.assertLess(relative_error, 1e-10)

                    reference = self.vertex_derivatives(
                        space, local_dof[cells[0]], cells[0], vertex, r[0]+1
                    )
                    for c in cells[1:]:
                        value = self.vertex_derivatives(
                            space, local_dof[c], c, vertex, r[0]+1
                        )
                        higher_order_jump = max(
                            higher_order_jump, bm.max(bm.abs(value-reference))
                        )

                self.assertGreater(higher_order_jump, 1e-6)


if __name__ == '__main__':
    unittest.main()
