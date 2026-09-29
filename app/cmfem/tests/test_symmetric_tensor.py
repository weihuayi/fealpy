import sys
import unittest
from itertools import permutations
from math import factorial, prod
from pathlib import Path

from fealpy.backend import backend_manager as bm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from symmetric_tensor import SymmetricTensor, symmetry_multiplicity


def assert_allclose(value, expected, atol):
    assert bool(bm.allclose(value, expected, atol=atol, rtol=1.0e-7))


def assert_array_equal(value, expected):
    value, expected = bm.asarray(value), bm.asarray(expected)
    assert value.shape == expected.shape
    assert bool(bm.all(value == expected))


class TestSymmetricTensor(unittest.TestCase):
    def tearDown(self):
        bm.set_backend('numpy')

    def test_inner(self):
        for backend in ('numpy', 'pytorch'):
            bm.set_backend(backend)
            for d, alpha, beta in [(2, (2, 0), (1, 1)),
                                   (3, (1, 2), (1, 1, 1)),
                                   (3, (2, 2), (0, 1, 3)),
                                   (2, (0, 0), (0,))]:
                with self.subTest(backend=backend, d=d, alpha=alpha, beta=beta):
                    shape = (4, len(alpha), d)
                    v = bm.sin(bm.arange(1, 1+4*len(alpha)*d,
                                         dtype=bm.float64)).reshape(shape)
                    w = bm.cos(bm.arange(1, 1+len(beta)*d,
                                         dtype=bm.float64)).reshape(len(beta), d)
                    a = SymmetricTensor(v, alpha)
                    b = SymmetricTensor(w, beta)
                    # Enumerate all low-order permutations as an independent reference.
                    left = [i for i, count in enumerate(alpha) for _ in range(count)]
                    right = [i for i, count in enumerate(beta) for _ in range(count)]
                    expected = bm.zeros(4, dtype=v.dtype)
                    for perm in permutations(right):
                        term = bm.ones(4, dtype=v.dtype)
                        for i, j in zip(left, perm):
                            term *= bm.sum(v[:, i]*w[j], axis=-1)
                        expected += term / factorial(sum(alpha))
                    assert_allclose(a.inner(b), expected, atol=1e-13)
                    assert_allclose(b.inner(a), expected, atol=1e-13)

    def test_dual_frames(self):
        bm.set_backend('numpy')
        frame = bm.array([[1., .2, .3], [.1, 1.2, -.2], [.3, .1, .9]])
        dual = bm.linalg.inv(frame).T
        indices = bm.multi_index_matrix(3, 2).tolist()
        for alpha in indices:
            for beta in indices:
                a, b = SymmetricTensor(frame, alpha), SymmetricTensor(dual, beta)
                expected = prod(factorial(i) for i in alpha)/factorial(3) if alpha == beta else 0
                assert_allclose(a.inner(b), expected, atol=1e-14)

    def test_batched_indices(self):
        for backend in ('numpy', 'pytorch'):
            bm.set_backend(backend)
            for d in (2, 3):
                vectors = bm.sin(bm.arange(1, 1+8*d, dtype=bm.float64))
                vectors = vectors.reshape(4, 2, d)
                other = bm.cos(bm.arange(1, 1+2*d, dtype=bm.float64))
                other = other.reshape(2, d)
                alpha = [(1, 2), (3, 0)]
                beta = [(0, 3), (2, 1), (1, 2)]
                a, b = SymmetricTensor(vectors, alpha), SymmetricTensor(other, beta)
                expected = bm.stack([bm.stack([
                    SymmetricTensor(vectors, x).inner(SymmetricTensor(other, y))
                    for y in beta], axis=-1) for x in alpha], axis=-2)
                assert_allclose(a.inner(b), expected, atol=1e-13)
                single = SymmetricTensor(other, beta[0])
                assert_allclose(a.inner(single), expected[..., 0], atol=1e-13)
                assert_allclose(single.inner(a), expected[..., 0], atol=1e-13)

    def test_symmetry_multiplicity(self):
        bm.set_backend('numpy')
        assert_array_equal(
            symmetry_multiplicity(3, 2), [1, 3, 3, 1]
        )
        assert_array_equal(
            symmetry_multiplicity(3, 3), [1, 3, 3, 3, 6, 3, 1, 3, 3, 1]
        )


if __name__ == '__main__':
    unittest.main()
