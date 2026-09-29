import itertools
import sys
import unittest
from math import factorial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import BernsteinFESpace
from fealpy.mesh import TetrahedronMesh
from cm_fe_space3d import CmDof3d, CmFESpace3d
from simplex_lattice import SimplexLattice


class TestCmDof3d(unittest.TestCase):
    def setUp(self):
        bm.set_backend('numpy')
        self.mesh = TetrahedronMesh.from_box(nx=1, ny=1, nz=1)

    def boundary_space(self, mesh, p, r):
        """Build only boundary geometry and numbering to avoid a large C2 matrix."""
        space = CmFESpace3d.__new__(CmFESpace3d)
        space.mesh = mesh
        space.p = p
        space.r = r
        space.ftype = mesh.ftype
        space.itype = bm.int32
        space.TD = 3
        space.GD = 3
        space.lattice = SimplexLattice(3, p, r)
        space.dof = CmDof3d(mesh, space.lattice)
        space.bspace = BernsteinFESpace(mesh, p=p, ctype='D')
        space._build_boundary_geometry()
        space.global_normal = space.global_normal_frame()
        space.vertex_frame = space.global_normal['node']
        space.edge_frame = space.global_normal['edge']
        space.face_frame = space.global_normal['face']
        return space

    def check_global_labels(self, mesh, p=9, r=(4, 2, 1, 0)):
        lattice = SimplexLattice(3, p, r)
        dof = CmDof3d(mesh, lattice)
        alpha = bm.tolist(lattice.multi_index)
        labels = []
        for dim, faces in enumerate(lattice.simplex.subsimplices):
            for entity_local, f in enumerate(faces):
                opposite = tuple(i for i in range(4) if i not in f)
                for s, rows in enumerate(lattice.layers[f]):
                    labels.extend((dim, entity_local, f, opposite, s, row)
                                  for row in bm.tolist(rows))
        self.assertEqual([label[-1] for label in labels], bm.tolist(lattice.permutation))

        cell = bm.tolist(mesh.entity('cell'))
        edge = bm.tolist(mesh.entity('edge'))
        face = bm.tolist(mesh.entity('face'))
        c2e = bm.tolist(mesh.cell_to_edge())
        c2f = bm.tolist(mesh.cell_to_face())
        cell_to_dof = bm.tolist(dof.cell_to_dof())
        ids_to_keys, keys_to_ids = {}, {}
        for c, vertices in enumerate(cell):
            for local, (dim, entity_local, f, opposite, s, row) in enumerate(labels):
                if dim == 0:
                    entity = vertices[f[0]]
                    tangent = (alpha[row][f[0]],)
                elif dim == 1:
                    entity = c2e[c][entity_local]
                    exponent = {vertices[i]: alpha[row][i] for i in f}
                    tangent = tuple(exponent[v] for v in edge[entity])
                elif dim == 2:
                    entity = c2f[c][entity_local]
                    exponent = {vertices[i]: alpha[row][i] for i in f}
                    tangent = tuple(exponent[v] for v in face[entity])
                else:
                    entity = c
                    tangent = tuple(alpha[row])
                normal = tuple(alpha[row][i] for i in opposite)
                key = (dim, entity, s, tangent, normal)
                gid = cell_to_dof[c][local]
                self.assertEqual(ids_to_keys.setdefault(gid, key), key)
                self.assertEqual(keys_to_ids.setdefault(key, gid), gid)

        self.assertEqual(len(ids_to_keys), dof.number_of_global_dofs())
        self.assertEqual(dof.cell_to_dof().shape,
                         (mesh.number_of_cells(), dof.number_of_local_dofs()))
        self.assertEqual(dof.cell_to_dof().dtype, bm.int32)

    def test_partition_counts_and_labels(self):
        parameters = [
            (1, (0, 0, 0, 0)),
            (9, (4, 2, 1, 0)),
            (11, (5, 2, 1, 0)),
            (13, (6, 3, 1, 0)),
        ]
        for p, r in parameters:
            with self.subTest(p=p, r=r):
                self.check_global_labels(self.mesh, p, r)
                lattice = SimplexLattice(3, p, r)
                dof = CmDof3d(self.mesh, lattice)
                nv, ne, nf, ni = [
                    len(lattice.indices(lattice.simplex.subsimplices[dim][0]))
                    for dim in range(4)
                ]
                self.assertEqual(4*nv + 6*ne + 4*nf + ni,
                                 (p+1)*(p+2)*(p+3)//6)
                self.assertEqual(dof.node_to_dof().shape,
                                 (self.mesh.number_of_nodes(), nv))
                self.assertEqual(dof.edge_to_internal_dof().shape,
                                 (self.mesh.number_of_edges(), ne))
                self.assertEqual(dof.face_to_internal_dof().shape,
                                 (self.mesh.number_of_faces(), nf))
                self.assertEqual(dof.cell_to_internal_dof().shape,
                                 (self.mesh.number_of_cells(), ni))

        dof = CmDof3d(self.mesh, SimplexLattice(3, 9, (4, 2, 1, 0)))
        self.assertEqual(dof.number_of_global_dofs(), 582)

    def test_shared_face_permutations(self):
        node = bm.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [0., 0., 1.], [0., 0., -1.]
        ])
        first = (0, 1, 2, 3)
        second = (0, 2, 1, 4)
        for permutation in itertools.permutations(range(4)):
            cell = bm.array([
                first, tuple(second[i] for i in permutation)
            ], dtype=bm.int32)
            self.check_global_labels(TetrahedronMesh(node, cell))

    def test_normal_frames(self):
        space = CmFESpace3d(self.mesh, 9, (4, 2, 1, 0))
        glambda = self.mesh.grad_lambda()
        for dim, faces in enumerate(space.lattice.simplex.subsimplices[:-1]):
            identity = bm.eye(3-dim, dtype=self.mesh.ftype)
            for f in faces:
                opposite = [i for i in range(4) if i not in f]
                product = bm.einsum(
                    'cid,cjd->cij', space.local_normal[f],
                    glambda[:, opposite, :]
                )
                self.assertTrue(bm.allclose(product, identity))

        point = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        tangent = point[edge[:, 1]] - point[edge[:, 0]]
        tangent = tangent / bm.sqrt(bm.sum(tangent**2, axis=1))[:, None]
        edge_frame = space.global_normal['edge']
        self.assertTrue(bm.allclose(
            bm.einsum('ed,eid->ei', tangent, edge_frame), 0
        ))
        self.assertTrue(bm.allclose(
            bm.einsum('eid,ejd->eij', edge_frame, edge_frame), bm.eye(2)
        ))

        face = self.mesh.entity('face')
        face_frame = space.global_normal['face'][:, 0]
        first = point[face[:, 1]] - point[face[:, 0]]
        second = point[face[:, 2]] - point[face[:, 0]]
        self.assertTrue(bm.allclose(bm.sum(face_frame*first, axis=1), 0))
        self.assertTrue(bm.allclose(bm.sum(face_frame*second, axis=1), 0))
        self.assertTrue(bm.allclose(bm.sum(face_frame**2, axis=1), 1))

    def test_local_dof_matrix(self):
        space = CmFESpace3d(self.mesh, 9, (4, 2, 1, 0))
        ldof = space.number_of_local_dofs()
        dense = space.D.to_dense()
        diagonal = bm.arange(ldof, dtype=bm.int32)

        self.assertEqual(space.D.shape,
                         (self.mesh.number_of_cells(), ldof, ldof))
        self.assertEqual(space.D.crow.dtype, bm.int32)
        self.assertEqual(space.D.col.dtype, bm.int32)
        self.assertTrue(bm.allclose(bm.triu(dense, 1), 0))
        self.assertTrue(bm.allclose(dense[:, diagonal, diagonal], 1))
        self.assertTrue(bm.all(
            bm.sum(space.lattice.multi_index, axis=1) == space.p
        ))

    def test_frame_transform_matrix(self):
        space = CmFESpace3d(self.mesh, 9, (4, 2, 1, 0))
        lattice = space.lattice
        ldof = space.number_of_local_dofs()
        dense = space.T.to_dense()
        entity_maps = (
            self.mesh.entity('cell'),
            self.mesh.cell_to_edge(),
            self.mesh.cell_to_face()
        )
        global_frames = (
            space.global_normal['node'],
            space.global_normal['edge'],
            space.global_normal['face']
        )

        self.assertEqual(space.T.shape,
                         (self.mesh.number_of_cells(), ldof, ldof))
        self.assertEqual(space.T.crow.dtype, bm.int32)
        self.assertEqual(space.T.col.dtype, bm.int32)

        for dim, faces in enumerate(lattice.simplex.subsimplices[:-1]):
            block_size = 3-dim
            for local_index, f in enumerate(faces):
                rows = lattice.layers[f][1][:block_size]
                index = lattice.inverse_permutation[rows]
                entity = entity_maps[dim][:, local_index]
                expected = bm.einsum(
                    'cid,cjd->cij',
                    space.local_normal[f],
                    space._dual_frame(global_frames[dim][entity])
                )
                self.assertTrue(bm.allclose(
                    dense[:, index[:, None], index[None, :]], expected
                ))

    def test_coefficient_and_interpolation(self):
        space = CmFESpace3d(self.mesh, 9, (4, 2, 1, 0))
        residual = bm.einsum(
            'cij,cjk->cik', space.D.to_dense(),
            bm.swapaxes(space.coeff, -1, -2)
        ) - space.T.to_dense()
        self.assertTrue(bm.allclose(residual, 0))

        direction = bm.array([1.0, 2.0, -0.5])
        degree = 4
        derivatives = []
        for s in range(5):
            alpha = bm.multi_index_matrix(s, 2, dtype=bm.int32)
            component = (
                factorial(degree)/factorial(degree-s)
                * bm.prod(direction[None, :]**alpha, axis=1)
            )

            def derivative(point, s=s, component=component):
                value = bm.einsum('...d,d->...', point, direction)
                if s == 0:
                    return value**degree
                return value[..., None]**(degree-s)*component

            derivatives.append(derivative)

        uh = space.interpolate(derivatives)
        bcs = bm.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25]
        ])
        point = self.mesh.bc_to_point(bcs)
        exact = bm.einsum('cqd,d->cq', point, direction)**degree
        self.assertTrue(bm.allclose(space.value(uh, bcs), exact))

    def test_boundary_geometry_and_frames(self):
        mesh = TetrahedronMesh.from_box(nx=2, ny=2, nz=2)
        space = self.boundary_space(mesh, 9, (4, 2, 1, 0))
        point = mesh.entity('node')

        boundary_face = bm.where(space.boundary_face_flag)[0]
        face = mesh.entity('face')[boundary_face]
        face_centroid = bm.sum(point[face], axis=1)/3
        adjacent = mesh.face_to_cell()[boundary_face, 0]
        cell_centroid = bm.sum(point[mesh.entity('cell')[adjacent]], axis=1)/4
        outward = bm.sum(
            space.face_frame[boundary_face, 0]
            * (cell_centroid-face_centroid), axis=1
        )
        self.assertTrue(bm.all(outward < 0))

        self.assertEqual(int(bm.sum(space.regular_boundary_edge_flag)), 48)
        self.assertEqual(int(bm.sum(space.ridge_edge_flag)), 24)
        self.assertEqual(int(bm.sum(space.regular_boundary_node_flag)), 6)
        self.assertEqual(int(bm.sum(space.ridge_node_flag)), 12)
        self.assertEqual(int(bm.sum(space.corner_node_flag)), 8)

        edge = mesh.entity('edge')
        tangent = point[edge[:, 1]]-point[edge[:, 0]]
        tangent = tangent/bm.sqrt(bm.sum(tangent**2, axis=1))[:, None]
        regular = bm.where(space.regular_boundary_edge_flag)[0]
        conormal = space.edge_frame[regular, 0]
        normal = space.edge_frame[regular, 1]
        self.assertTrue(bm.allclose(bm.sum(tangent[regular]*conormal, axis=1), 0))
        self.assertTrue(bm.allclose(bm.sum(tangent[regular]*normal, axis=1), 0))
        self.assertTrue(bm.allclose(bm.sum(conormal*normal, axis=1), 0))
        self.assertTrue(bm.allclose(bm.cross(tangent[regular], conormal), normal))

        for edge_id in bm.tolist(bm.where(space.ridge_edge_flag)[0]):
            face_ids = space._edge_boundary_faces[edge_id]
            product = bm.einsum(
                'id,jd->ij', space.edge_frame[edge_id],
                space.boundary_face_normal[face_ids]
            )
            self.assertTrue(bm.allclose(product, bm.eye(2)))

        for node_id in bm.tolist(bm.where(space.ridge_node_flag)[0]):
            face_ids = space._node_boundary_faces[node_id]
            product = bm.einsum(
                'id,jd->ij', space.vertex_frame[node_id, 1:],
                space.boundary_face_normal[face_ids]
            )
            self.assertTrue(bm.allclose(product, bm.eye(2)))

    def test_boundary_masks(self):
        mesh = TetrahedronMesh.from_box(nx=2, ny=2, nz=2)
        parameters = [
            (9, (4, 2, 1, 0), [1, 2, 2], [0, 0, 0, 0, 1]),
            (17, (8, 4, 2, 0), [1, 2, 3, 3, 3],
             [0, 0, 0, 0, 0, 0, 1, 3, 6]),
        ]
        for p, r, edge_components, ridge_free in parameters:
            with self.subTest(p=p):
                space = self.boundary_space(mesh, p, r)
                boundary = space.is_boundary_dof()

                face_dof = space.dof.face_to_internal_dof()
                self.assertTrue(bm.all(boundary[face_dof[space.boundary_face_flag]]))

                edge_face = space.lattice.simplex.subsimplices[1][0]
                regular_edge = int(bm.where(space.regular_boundary_edge_flag)[0][0])
                ridge_edge = int(bm.where(space.ridge_edge_flag)[0][0])
                start = 0
                for q, rows in enumerate(space.lattice.layers[edge_face]):
                    size = len(rows)
                    if size:
                        regular_dof = space.dof.edge_to_internal_dof()[
                            regular_edge, start:start+size
                        ]
                        ridge_dof = space.dof.edge_to_internal_dof()[
                            ridge_edge, start:start+size
                        ]
                        tangent_count = size//(q+1)
                        self.assertEqual(
                            int(bm.sum(boundary[regular_dof])),
                            tangent_count*edge_components[q]
                        )
                        self.assertTrue(bm.all(boundary[ridge_dof]))
                    start += size

                node_face = space.lattice.simplex.subsimplices[0][0]
                ridge_node = int(bm.where(space.ridge_node_flag)[0][0])
                corner_node = int(bm.where(space.corner_node_flag)[0][0])
                start = 0
                for q, rows in enumerate(space.lattice.layers[node_face]):
                    size = len(rows)
                    ridge_dof = space.dof.node_to_dof()[
                        ridge_node, start:start+size
                    ]
                    corner_dof = space.dof.node_to_dof()[
                        corner_node, start:start+size
                    ]
                    self.assertEqual(
                        int(bm.sum(~boundary[ridge_dof])), ridge_free[q]
                    )
                    self.assertTrue(bm.all(boundary[corner_dof]))
                    start += size

    def test_skew_ridge_transform_and_polynomial(self):
        node = bm.array([
            [0.0, 0.0, 0.0], [1.2, 0.1, 0.0],
            [0.2, 1.1, 0.2], [0.1, 0.3, 1.4]
        ])
        mesh = TetrahedronMesh(node, bm.array([[0, 1, 2, 3]], dtype=bm.int32))
        space = CmFESpace3d(mesh, 9, (4, 2, 1, 0))
        residual = bm.einsum(
            'cij,cjk->cik', space.D.to_dense(),
            bm.swapaxes(space.coeff, -1, -2)
        )-space.T.to_dense()
        self.assertLess(float(bm.max(bm.abs(residual))), 1.0e-12)

        direction = bm.array([1.0, -0.7, 0.4])
        degree = 4
        derivatives = []
        for s in range(5):
            alpha = bm.multi_index_matrix(s, 2, dtype=bm.int32)
            component = (
                factorial(degree)/factorial(degree-s)
                * bm.prod(direction[None, :]**alpha, axis=1)
            )

            def derivative(point, s=s, component=component):
                value = bm.einsum('...d,d->...', point, direction)
                if s == 0:
                    return value**degree
                return value[..., None]**(degree-s)*component

            derivatives.append(derivative)

        uh = space.interpolate(derivatives)
        bcs = bm.array([[0.1, 0.2, 0.3, 0.4]])
        exact = bm.einsum('cqd,d->cq', mesh.bc_to_point(bcs), direction)**degree
        self.assertTrue(bm.allclose(space.value(uh, bcs), exact, atol=1.0e-12))

    def test_boundary_interpolation(self):
        mesh = TetrahedronMesh.from_box(nx=2, ny=2, nz=2)
        space = self.boundary_space(mesh, 9, (4, 2, 1, 0))
        direction = bm.array([1.0, 0.3, -0.6])
        degree = 4
        derivatives = []
        for s in range(5):
            alpha = bm.multi_index_matrix(s, 2, dtype=bm.int32)
            component = (
                factorial(degree)/factorial(degree-s)
                * bm.prod(direction[None, :]**alpha, axis=1)
            )

            def derivative(point, s=s, component=component):
                value = bm.einsum('...d,d->...', point, direction)
                if s == 0:
                    return value**degree
                return value[..., None]**(degree-s)*component

            derivatives.append(derivative)

        interpolant = space.interpolate(derivatives)
        initial = bm.arange(space.number_of_global_dofs(), dtype=space.ftype)
        boundary_value, boundary = space.boundary_interpolate(
            derivatives, bm.copy(initial)
        )
        self.assertTrue(bm.allclose(
            boundary_value[boundary], interpolant[boundary]
        ))
        self.assertTrue(bm.allclose(
            boundary_value[~boundary], initial[~boundary]
        ))


if __name__ == '__main__':
    unittest.main()
