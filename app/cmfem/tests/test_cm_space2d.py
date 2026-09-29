import sys
import unittest
from math import factorial
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cm_fe_space2d import CmFESpace2d
from sparse_tensor_ops import spmv, spsolve_triangular


def assert_allclose(value, expected, rtol=1.0e-7, atol=0.0):
    assert bool(bm.allclose(value, expected, rtol=rtol, atol=atol))


def assert_array_equal(value, expected):
    value, expected = bm.asarray(value), bm.asarray(expected)
    assert value.shape == expected.shape or expected.ndim == 0
    assert bool(bm.all(value == expected))


def derivative_dof_matrix(space, global_frame=False):
    """Differentiate every Bernstein polynomial to verify normalized DoFs."""
    p = space.p
    mesh, lattice = space.mesh, space.lattice
    indices = [bm.multi_index_matrix(q, 2) for q in range(p+1)]
    lookup = [{tuple(a): i for i, a in enumerate(idx)} for idx in indices]
    nc, ndof = mesh.number_of_cells(), space.number_of_local_dofs()
    result = bm.zeros((nc, ndof, ndof))
    gradients = mesh.grad_lambda()
    points = mesh.entity('node')[mesh.entity('cell')]
    row = 0
    for ell, faces in enumerate(lattice.simplex.subsimplices):
        for f in faces:
            opposite = [i for i in range(3) if i not in f]
            for s, rows in enumerate(lattice.layers[f]):
                for alpha in lattice.multi_index[rows]:
                    for c in range(nc):
                        coefficients = bm.eye(ndof)
                        q = p
                        for normal_index, j in enumerate(opposite):
                            if global_frame and ell == 0:
                                vertex = mesh.entity('cell')[c, f[0]]
                                normal = space.global_normal['node'][vertex, normal_index]
                            elif global_frame:
                                edge_index = lattice.simplex.subsimplices[1].index(f)
                                edge = mesh.cell_to_edge()[c, edge_index]
                                normal = space.global_normal['edge'][edge, 0]
                            elif ell == 0:
                                normal = points[c, j] - points[c, f[0]]
                            else:
                                normal = gradients[c, j] / bm.dot(gradients[c, j], gradients[c, j])
                            for _ in range(int(alpha[j])):
                                derivative = bm.zeros((len(indices[q-1]), ndof))
                                for a, gamma in enumerate(indices[q-1]):
                                    for v in range(3):
                                        beta = gamma.copy()
                                        beta[v] += 1
                                        derivative[a] += q*bm.dot(normal, gradients[c, v])*coefficients[lookup[q][tuple(beta)]]
                                coefficients = derivative
                                q -= 1
                        target = tuple(int(alpha[i]) if i in f else 0 for i in range(3))
                        result[c, row] = coefficients[lookup[q][target]][lattice.permutation] * factorial(p-s)/factorial(p)
                    row += 1
    return result


class TestCmFESpace2d(unittest.TestCase):
    def global_coefficients(self, space, bernstein):
        """Recover consistent global DoFs from cellwise Bernstein coefficients."""
        reordered = bernstein[:, space.lattice.permutation]
        local_dof = spmv(space.D, reordered)
        local_dof = bm.linalg.solve(
            space.T.to_dense(), local_dof[..., None]
        )[..., 0]
        uh = bm.zeros(space.number_of_global_dofs())
        assigned = {}
        for dofs, values in zip(space.cell_to_dof(), local_dof):
            for dof, value in zip(dofs, values):
                if int(dof) in assigned:
                    assert_allclose(value, assigned[int(dof)], atol=1e-11)
                else:
                    assigned[int(dof)] = value
                    uh[int(dof)] = value
        return uh

    def test_stiffness_matrix(self):
        bm.set_backend('numpy')
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)

        # u=xy: |nabla^2 u|_F^2=2 checks the mixed second-derivative multiplicity.
        space = CmFESpace2d(mesh, 5, (2, 1, 0))
        alpha = space.lattice.multi_index
        cells = mesh.entity('cell')
        points = mesh.entity('node')
        bernstein = []
        for vertices in cells:
            x, y = points[vertices, 0], points[vertices, 1]
            coefficient = ((alpha@x)*(alpha@y)-alpha@(x*y))/(space.p*(space.p-1))
            bernstein.append(coefficient)
        uh = self.global_coefficients(space, bm.stack(bernstein))
        matrix = space.stiffness_matrix()
        dense = matrix.to_dense()
        assert_allclose(dense, dense.T, atol=1e-12)
        energy = bm.einsum('i,i->', uh, matrix@uh)
        assert_allclose(energy, 2*bm.sum(mesh.entity_measure('cell')), rtol=1e-12)

        # u=x^9 checks the third-order polyharmonic energy form.
        space = CmFESpace2d(mesh, 9, (4, 2, 0))
        alpha = space.lattice.multi_index
        bernstein = []
        for vertices in cells:
            x = points[vertices, 0]
            bernstein.append(bm.prod(x[None, :]**alpha, axis=1))
        uh = self.global_coefficients(space, bm.stack(bernstein))
        matrix = space.stiffness_matrix()
        energy = bm.einsum('i,i->', uh, matrix@uh)
        exact = (9*8*7)**2/13
        assert_allclose(energy, exact, rtol=2e-11)

    def test_interpolation_and_source_vector(self):
        bm.set_backend('numpy')
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
        space = CmFESpace2d(mesh, 5, (2, 1, 0))

        def value(point):
            x, y = point[..., 0], point[..., 1]
            return x**3*y**2 + 2*x

        def gradient(point):
            x, y = point[..., 0], point[..., 1]
            return bm.stack((3*x**2*y**2+2, 2*x**3*y), axis=-1)

        def hessian(point):
            x, y = point[..., 0], point[..., 1]
            return bm.stack((6*x*y**2, 6*x**2*y, 2*x**3), axis=-1)

        uh = space.interpolate([value, gradient, hessian])
        bcs = bm.array([[.2, .3, .5], [.6, .1, .3]])
        point = mesh.bc_to_point(bcs)
        assert_allclose(space.value(uh, bcs), value(point), atol=2e-14)
        assert_allclose(space.grad_value(uh, bcs), gradient(point), atol=3e-14)
        assert_allclose(space.grad_m_value(uh, bcs, 2), hessian(point), atol=1e-13)

        one = lambda point: bm.ones(point.shape[:-1])
        zero1 = lambda point: bm.zeros(point.shape[:-1] + (2,))
        zero2 = lambda point: bm.zeros(point.shape[:-1] + (3,))
        constant = space.interpolate([one, zero1, zero2])
        load = space.source_vector(one)
        assert_allclose(load @ constant, 1.0, atol=2e-14)

    def test_boundary_frames_dofs_and_interpolation(self):
        bm.set_backend('numpy')
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
        point = mesh.entity('node')

        for p, r, expected in [
                (5, (2, 1, 0), (1, 2, 2)),
                (9, (4, 2, 0), (1, 2, 3, 3, 3))]:
            with self.subTest(r=r):
                space = CmFESpace2d(mesh, p, r)
                regular = space.regular_boundary_node_flag
                corner = space.corner_node_flag
                interior = space.interior_node_flag
                assert_array_equal(regular, bm.array([
                    False, True, False, True, False, True, False, True, False
                ]))
                assert_array_equal(corner, bm.array([
                    True, False, True, False, False, False, True, False, True
                ]))
                assert_array_equal(interior, bm.array([
                    False, False, False, False, True, False, False, False, False
                ]))

                frame = space.vertex_frame[regular]
                assert_allclose(
                    bm.einsum('nid,njd->nij', frame, frame),
                    bm.tile(bm.eye(2), (len(frame), 1, 1)), atol=1e-14
                )

                edge = mesh.entity('edge')
                edge_midpoint = point[edge].mean(axis=1)
                normal = space.global_normal['edge'][:, 0]
                for midpoint, value, is_boundary in zip(
                        edge_midpoint, normal, space.boundary_edge_flag):
                    if not is_boundary:
                        continue
                    if bm.abs(midpoint[0]) < 1.0e-12:
                        exact = (-1, 0)
                    elif bm.abs(midpoint[0]-1) < 1.0e-12:
                        exact = (1, 0)
                    elif bm.abs(midpoint[1]) < 1.0e-12:
                        exact = (0, -1)
                    else:
                        exact = (0, 1)
                    assert_allclose(value, exact, atol=1e-14)

                boundary = space.is_boundary_dof()
                node_boundary = boundary[space.dof.node_to_dof()]
                regular_node = bm.where(regular)[0][0]
                corner_node = bm.where(corner)[0][0]
                start = 0
                for q, rows in enumerate(space.lattice.layers[(0,)]):
                    size = len(rows)
                    self.assertEqual(node_boundary[regular_node, start:start+size].sum(),
                                     expected[q])
                    self.assertTrue(node_boundary[corner_node, start:start+size].all())
                    start += size
                edge_boundary = boundary[space.dof.edge_to_internal_dof()]
                self.assertTrue(edge_boundary[space.boundary_edge_flag].all())
                self.assertFalse(edge_boundary[~space.boundary_edge_flag].any())

                def derivative(s):
                    def value(x):
                        power_x, power_y = r[0], p-r[0]
                        terms = []
                        for j in range(s+1):
                            a, b = s-j, j
                            coefficient = bm.prod(
                                bm.arange(power_x-a+1, power_x+1)
                            )*bm.prod(bm.arange(power_y-b+1, power_y+1)) / (
                                factorial(power_x)*factorial(power_y)
                            )
                            terms.append(coefficient*x[..., 0]**(power_x-a)
                                         * x[..., 1]**(power_y-b))
                        return terms[0] if s == 0 else bm.stack(terms, axis=-1)
                    return value

                derivatives = [derivative(s) for s in range(r[0]+1)]
                all_value = space.interpolate(derivatives)
                bcs = bm.array([[.2, .3, .5], [.6, .1, .3]])
                physical_point = mesh.bc_to_point(bcs)
                for q in range(r[0]+1):
                    value = (space.value(all_value, bcs) if q == 0
                             else space.grad_m_value(all_value, bcs, q))
                    assert_allclose(
                        value, derivatives[q](physical_point), atol=5e-10
                    )
                bm.random.seed(42)
                initial = bm.random.normal(size=space.number_of_global_dofs())
                boundary_value, flag = space.boundary_interpolate(
                    derivatives, initial
                )
                assert_allclose(boundary_value[flag], all_value[flag], atol=1e-13)
                assert_array_equal(boundary_value[~flag], initial[~flag])

                D = space.D.to_dense()
                T = space.T.to_dense()
                G = derivative_dof_matrix(space, global_frame=True)
                assert_allclose(T @ G, D, atol=5e-13)
                duality = G @ bm.swapaxes(space.coeff, 1, 2)
                assert_allclose(
                    duality,
                    bm.tile(bm.eye(space.number_of_local_dofs()),
                            (mesh.number_of_cells(), 1, 1)),
                    atol=2e-11
                )

        general = CmFESpace2d(mesh, 7, (3, 1, 0))
        with self.assertRaises(NotImplementedError):
            general.is_boundary_dof()

    def test_value_and_derivatives(self):
        bm.set_backend('numpy')
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
        space = CmFESpace2d(mesh, 5, (2, 1, 0))
        bcs = bm.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
        bm.random.seed(17)
        uh = bm.random.normal(size=space.number_of_global_dofs())
        local = uh[space.cell_to_dof()]

        phi = space.basis(bcs)
        expected_value = bm.einsum('cqi,ci->cq', phi, local)
        assert_allclose(space.value(uh, bcs), expected_value)
        assert_allclose(space.function(array=uh)(bcs), expected_value)

        for m in range(1, 4):
            phi = space.grad_m_basis(bcs, m)
            expected = bm.einsum('cqig,ci->cqg', phi, local)
            assert_allclose(space.grad_m_value(uh, bcs, m), expected)
        assert_allclose(space.grad_value(uh, bcs), space.grad_m_value(uh, bcs, 1))

        batch = bm.stack((uh, 2*uh), axis=0)
        assert_allclose(
            space.value(batch, bcs), bm.stack((expected_value, 2*expected_value))
        )

    def test_normal_frames(self):
        bm.set_backend('numpy')
        # Include horizontal, vertical, diagonal, and general edge directions.
        node = bm.array([[0., 0.], [1., 0.], [0., 1.], [1.4, 1.2]])
        cell = bm.array([[0, 1, 2], [3, 2, 1]], dtype=bm.int32)
        space = CmFESpace2d(TriangleMesh(node, cell), 5, (2, 1, 0))
        mesh = space.mesh
        glambda = mesh.grad_lambda()
        for f, normal in space.local_normal.items():
            opposite = [j for j in range(3) if j not in f]
            pairing = bm.einsum('cid,cjd->cij', normal, glambda[:, opposite])
            expected = bm.broadcast_to(bm.eye(len(opposite)), pairing.shape)
            assert_allclose(pairing, expected, atol=1e-14)
        assert_array_equal(space.global_normal['node'], bm.tile(bm.eye(2), (4, 1, 1)))
        edge = mesh.entity('edge')
        normal = space.global_normal['edge'][:, 0]
        tangent = node[edge[:, 1]] - node[edge[:, 0]]
        assert_allclose(bm.sum(normal*tangent, axis=1), 0, atol=1e-14)
        assert_allclose(bm.sum(normal**2, axis=1), 1, atol=1e-14)
        pivot = bm.argmax(bm.abs(normal), axis=1)
        interior = ~mesh.boundary_edge_flag()
        self.assertTrue(bm.all(normal[bm.arange(len(edge))[interior], pivot[interior]] > 0))
        # Permuting cell vertices must leave each global edge frame unchanged.
        other = CmFESpace2d(TriangleMesh(node, cell[:, [1, 2, 0]]), 5, (2, 1, 0))
        frames = {tuple(sorted(e)): n for e, n in zip(edge.tolist(), normal)}
        for e, n in zip(other.mesh.entity('edge').tolist(), other.global_normal['edge'][:, 0]):
            assert_allclose(n, frames[tuple(sorted(e))], atol=1e-14)

    def test_local_dof_matrix(self):
        bm.set_backend('numpy')
        node = bm.array([[0., 0.], [1.3, .2], [.1, 1.1], [1.5, 1.4]])
        cell = bm.array([[0, 1, 2], [3, 2, 1]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)
        for p, r in [(1, (0, 0, 0)), (5, (2, 1, 0)),
                     (7, (3, 1, 0)), (9, (4, 2, 0))]:
            with self.subTest(p=p, r=r):
                space = CmFESpace2d(mesh, p, r)
                D = space.D
                dense = D.to_dense()
                assert_allclose(dense, derivative_dof_matrix(space), atol=2e-13)
                assert_array_equal(bm.triu(dense, 1), 0)
                assert_array_equal(bm.einsum('cii->ci', dense), 1)
                self.assertEqual(D.col.dtype, bm.int32)
                self.assertEqual(D.crow.dtype, bm.int32)
                bm.random.seed(42)
                x = bm.random.normal(size=dense.shape[:2])
                rhs = spmv(D, x)
                assert_allclose(rhs, bm.einsum('cij,cj->ci', dense, x), atol=1e-13)
                assert_allclose(spsolve_triangular(D, rhs, unit_diagonal=True), x, atol=1e-12)

    def test_frame_transform_matrix(self):
        bm.set_backend('numpy')
        node = bm.array([[0., 0.], [1.3, .2], [.1, 1.1], [1.5, 1.4]])
        cell = bm.array([[0, 1, 2], [3, 2, 1]], dtype=bm.int32)
        for p, r in [(1, (0, 0, 0)), (5, (2, 1, 0)),
                     (7, (3, 1, 0)), (9, (4, 2, 0))]:
            with self.subTest(p=p, r=r):
                space = CmFESpace2d(TriangleMesh(node, cell), p, r)
                T = space.T.to_dense()
                G = derivative_dof_matrix(space, global_frame=True)
                D = space.D.to_dense()
                assert_allclose(T @ G, D, atol=3e-13)
                C = space.coeff
                assert_allclose(D @ C.transpose(0, 2, 1), T, atol=3e-13)
                assert_allclose(
                    C, bm.linalg.solve(D, T).transpose(0, 2, 1), atol=3e-13
                )
                self.assertEqual(space.T.col.dtype, bm.int32)
                self.assertEqual(space.T.crow.dtype, bm.int32)
                # Shared DoFs must agree for the same global polynomial x^p.
                vertices = node[cell, 0]
                alpha = space.lattice.multi_index[space.lattice.permutation]
                coefficients = bm.prod(vertices[:, None, :]**alpha[None, :, :], axis=2)
                local_values = spmv(space.D, coefficients)
                global_values = bm.linalg.solve(T, local_values[..., None])[..., 0]
                seen = {}
                for ids, values in zip(space.cell_to_dof(), global_values):
                    for dof, value in zip(ids, values):
                        if int(dof) in seen:
                            assert_allclose(value, seen[int(dof)], atol=1e-12)
                        seen[int(dof)] = value


if __name__ == '__main__':
    unittest.main()
