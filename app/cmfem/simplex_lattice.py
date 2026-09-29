from itertools import combinations

from fealpy.backend import backend_manager as bm


class Simplex:
    """Reference d-simplex with unoriented, ascending local subsimplices."""

    def __init__(self, d):
        self.d = d
        self.vertices = tuple(range(d+1))
        self.subsimplices = []
        for ell in range(d+1):
            faces = tuple(combinations(self.vertices, ell+1))
            self.subsimplices.append(faces)
        if d == 2:
            self.subsimplices[1] = ((1, 2), (2, 0), (0, 1))
        elif d == 3:
            self.subsimplices[2] = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
        self.subsimplices = tuple(self.subsimplices)


class SimplexLattice:
    """
    Decomposition of T_k^d for d>0, r[l]>=2*r[l+1]>=0, r[d]=0, and k>=2*r[0]+1.
    """

    def __init__(self, d, k, r):
        self._is_valid(d, k, r)

        self.d, self.k, self.r = d, k, r
        self.simplex = Simplex(d)

        self.multi_index = bm.multi_index_matrix(k, d, dtype=bm.int32)

        alpha = self.multi_index
        used = bm.zeros(alpha.shape[0], dtype=bm.bool, device=bm.get_device(alpha))  # Multi-indices already assigned to a subsimplex.
        self.layers = {}  # Multi-index rows in every layer of each subsimplex.
        blocks = []       # Layer blocks ordered by increasing simplex dimension.
        for ell, faces in enumerate(self.simplex.subsimplices):
            for f in faces:
                distance = k-bm.sum(alpha[:, list(f)], axis=1)

                layers = [bm.astype(bm.where((distance == s) & ~used)[0], bm.int32)
                          for s in range(r[ell]+1)]
                # Sort each layer first by tangential and then by normal indices.
                opposite = tuple(i for i in self.simplex.vertices if i not in f)
                order = f + opposite
                for s, rows in enumerate(layers):
                    keys = tuple(-alpha[rows, i] for i in reversed(order))
                    layers[s] = rows[bm.lexsort(keys)]
                self.layers[f] = tuple(layers)
                blocks.extend(layers)
                used = bm.set_at(used, bm.concatenate(layers), True)

        self.permutation = bm.concatenate(blocks)
        self.inverse_permutation = bm.astype(bm.argsort(self.permutation), bm.int32)

    def indices(self, f, s=None):
        """Return original row indices of a block or layer ``s``."""
        layers = self.layers[f]
        return bm.concatenate(layers) if s is None else layers[s]

    def _is_valid(self, d, k, r):
        """Check d>0, r[l]>=2*r[l+1]>=0, r[d]=0, and k>=2*r[0]+1."""
        if d <= 0 or len(r) != d+1 or r[-1] != 0:
            raise ValueError("Require d>0, r=(r0,...,rd), rd=0.")
        for l in range(d):
            if r[l] < 2*r[l+1] or r[l] < 0:
                raise ValueError("Require r[l]>=2*r[l+1]>=0.")
        if k < 2*r[0]+1:
            raise ValueError("Require k>=2*r[0]+1.")
