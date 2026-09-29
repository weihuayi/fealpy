import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from cm_fe_space2d import CmDof2d
from simplex_lattice import SimplexLattice


class TestCmDof2d(unittest.TestCase):
    def setUp(self):
        bm.set_backend("numpy")
        self.mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)

    def test_partition_and_counts(self):
        for r1 in range(3):
            for r0 in range(2*r1, 2*r1+4):
                for k in (2*r0+1, 2*r0+3):
                    lattice = SimplexLattice(2, k, (r0, r1, 0))
                    dof = CmDof2d(self.mesh, lattice)
                    self.assertIs(dof.lattice, lattice)
                    self.check_global_labels(self.mesh, lattice)
                    nv = (r0+1)*(r0+2)//2
                    ne = (r1+1)*(k-2*r0-1)+r1*(r1+1)//2
                    ni = (k+1)*(k+2)//2-3*nv-3*ne
                    self.assertEqual(dof.node_to_dof().shape, (4, nv))
                    self.assertEqual(dof.edge_to_internal_dof().shape, (5, ne))
                    self.assertEqual(dof.cell_to_internal_dof().shape, (2, ni))
                    self.assertEqual(dof.number_of_global_dofs(), 4*nv+5*ne+2*ni)

    def check_global_labels(self, mesh, lattice=None):
        if lattice is None:
            lattice = SimplexLattice(2, 11, (4, 1, 0))
        dof = CmDof2d(mesh, lattice)
        alpha = lattice.multi_index.tolist()
        # Build the label bijection independently and verify every permuted column.
        labels = []
        for faces in lattice.simplex.subsimplices:
            for f in faces:
                for order, layer in enumerate(lattice.layers[f]):
                    labels.extend((f, order, row) for row in layer.tolist())
        self.assertEqual([row for _, _, row in labels], lattice.permutation.tolist())
        ids_to_keys, keys_to_ids = {}, {}
        edge_lookup = {frozenset(edge): tuple(edge) for edge in mesh.entity('edge').tolist()}
        for c, vertices in enumerate(mesh.cell.tolist()):
            for row, (f, order, index) in enumerate(labels):
                a = alpha[index]
                if len(f) == 1:
                    normal = tuple(a[j] for j in range(3) if j not in f)
                    key = (0, vertices[f[0]], normal)
                elif len(f) == 2:
                    tangent = tuple(sorted((vertices[j], a[j]) for j in f))
                    key = (1, tangent, order)
                    # The second endpoint exponent increases along the mesh edge direction.
                    edge = edge_lookup[frozenset(vertices[j] for j in f)]
                    second_exponent = dict(tangent)[edge[1]]
                    gid = int(dof.cell_to_dof()[c, row])
                    rank = sum(lattice.k-2*lattice.r[0]+t-1 for t in range(order))
                    rank += second_exponent-(lattice.r[0]-order+1)
                    nv = len(lattice.indices((0,)))
                    ne = len(lattice.indices((0, 1)))
                    self.assertEqual((gid-mesh.number_of_nodes()*nv) % ne, rank)
                else:
                    key = (2, c, tuple(a))
                gid = int(dof.cell_to_dof()[c, row])
                self.assertEqual(ids_to_keys.setdefault(gid, key), key)
                self.assertEqual(keys_to_ids.setdefault(key, gid), gid)
        self.assertEqual(len(ids_to_keys), dof.number_of_global_dofs())
        self.assertEqual(dof.cell_to_dof().shape,
                         (mesh.number_of_cells(), dof.number_of_local_dofs()))

    def test_vertex_permutations(self):
        cells = self.mesh.cell.copy()
        for p in itertools.permutations(range(3)):
            for q in itertools.permutations(range(3)):
                cell = bm.stack((cells[0, list(p)], cells[1, list(q)]))
                self.check_global_labels(TriangleMesh(self.mesh.node, cell))

    def test_pytorch_tensors(self):
        # Reuse topology: the installed FEALPy's PyTorch mesh builder currently
        # requires an unavailable cumulative_sum backend method.
        mesh = self.mesh
        cell, local_edge, c2e = mesh.cell, mesh.localEdge, mesh.cell_to_edge()
        entities = {'cell': cell, 'edge': mesh.entity('edge')}
        bm.set_backend("pytorch")
        try:
            proxy = SimpleNamespace(
                cell=bm.array(cell, dtype=bm.int32),
                entity=lambda etype: bm.array(entities[etype], dtype=bm.int32),
                localEdge=bm.array(local_edge, dtype=bm.int32),
                cell_to_edge=lambda: bm.array(c2e, dtype=bm.int32),
                top_dimension=lambda: 2, geo_dimension=lambda: 2,
                number_of_nodes=lambda: 4, number_of_edges=lambda: 5,
                number_of_cells=lambda: 2)
            self.check_global_labels(proxy)
            for k, r in ((1, (0, 0, 0)), (5, (2, 1, 0))):
                dof = CmDof2d(proxy, SimplexLattice(2, k, r))
                self.assertEqual(dof.cell_to_dof().shape[1], (k+1)*(k+2)//2)
        finally:
            bm.set_backend("numpy")

if __name__ == "__main__":
    unittest.main()
