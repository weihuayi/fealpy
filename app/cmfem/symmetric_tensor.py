"""Symmetrized vector products and their tensor inner products."""

from math import factorial, prod

from fealpy.backend import backend_manager as bm


def symmetry_multiplicity(order, dimension, dtype=None):
    """Return the multiplicities ``order!/alpha!`` of symmetric components."""
    alpha = bm.multi_index_matrix(order, dimension-1, dtype=bm.int32)
    values = [factorial(order)//prod(factorial(a) for a in row)
              for row in bm.tolist(alpha)]
    return bm.array(values, dtype=dtype)


class SymmetricTensor:
    """Represent ``sym(v_0^alpha_0 tensor ... tensor v_{m-1}^alpha_{m-1})``.

    ``vectors`` has shape ``(..., m, d)`` with one vector per row.
    ``alpha`` is one nonnegative multi-index of length ``m`` or an array
    of equal-order multi-indices with shape ``(NA, m)``. Batched inner
    products have shape ``(..., NA, NB)``. Full ``d**s`` component arrays
    are never formed.
    """

    def __init__(self, vectors, alpha):
        self.vectors = vectors
        indices = bm.array(alpha, dtype=bm.int32)
        self.batched = indices.ndim == 2
        self.alpha = (tuple(tuple(a) for a in bm.tolist(indices)) if self.batched
                      else tuple(bm.tolist(indices)))
        self.order = sum(self.alpha[0] if self.batched else self.alpha)

    def inner(self, other):
        """Contract equal-order symmetric tensors with broadcast batch axes.

        The contraction averages over permutations of the second tensor.
        Repeated states are merged by direction counts, avoiding explicit
        enumeration of all ``s!`` permutations.
        """
        if self.order != other.order:
            raise ValueError("Tensor orders must match.")
        if self.batched or other.batched:
            left = self.alpha if self.batched else (self.alpha,)
            right = other.alpha if other.batched else (other.alpha,)
            if self.vectors.shape[-2:] == (2, 2) and other.vectors.shape[-2:] == (2, 2):
                # Contract every pair of 2D frame multi-indices in one batch.
                a, multiplicity = symmetric_span_array(self.vectors, self.order)
                b, _ = symmetric_span_array(other.vectors, self.order)
                ai = bm.array([alpha[1] for alpha in left], dtype=bm.int32,
                              device=bm.get_device(a))
                bi = bm.array([beta[1] for beta in right], dtype=bm.int32,
                              device=bm.get_device(b))
                result = bm.einsum(
                    '...aq,...bq,q->...ab',
                    a[..., ai, :], b[..., bi, :], multiplicity
                )
            else:
                result = bm.stack([
                    bm.stack([SymmetricTensor(self.vectors, alpha).inner(
                        SymmetricTensor(other.vectors, beta)) for beta in right], axis=-1)
                    for alpha in left], axis=-2)
            if not self.batched:
                result = result[..., 0, :]
            if not other.batched:
                result = result[..., 0]
            return result
        if len(self.alpha) == len(other.alpha) == 1:
            return bm.sum(self.vectors[..., 0, :]*other.vectors[..., 0, :], axis=-1)**self.order
        gram = bm.einsum('...id,...jd->...ij', self.vectors, other.vectors)
        zero = (0,)*len(other.alpha)
        states = {zero: bm.ones(gram.shape[:-2], dtype=gram.dtype,
                               device=bm.get_device(gram))}
        remaining = self.order
        for i, count in enumerate(self.alpha):
            for _ in range(count):
                next_states = {}
                for used, value in states.items():
                    for j, total in enumerate(other.alpha):
                        left = total - used[j]
                        if left == 0:
                            continue
                        target = list(used)
                        target[j] += 1
                        target = tuple(target)
                        term = value * gram[..., i, j] * (left / remaining)
                        next_states[target] = next_states.get(target, 0) + term
                states = next_states
                remaining -= 1
        return states[other.alpha]


def symmetric_span_array(frame, s):
    """Return independent components of every ``sym(frame**alpha)``.

    ``frame`` has shape ``(..., m, d)`` with one direction per row. The
    penultimate output axis indexes the ``m``-dimensional multi-indices,
    and the last axis indexes the symmetric components in dimension ``d``.
    """
    if frame.shape[-2:] != (2, 2):
        m, d = frame.shape[-2:]
        alpha = bm.multi_index_matrix(s, m-1, dtype=bm.int32)
        component = bm.multi_index_matrix(s, d-1, dtype=bm.int32)
        identity = bm.eye(d, dtype=frame.dtype)
        tensor = SymmetricTensor(frame, alpha).inner(
            SymmetricTensor(identity, component)
        )
        multiplicity = symmetry_multiplicity(s, d, dtype=frame.dtype)
        return tensor, multiplicity

    coefficients = bm.ones(frame.shape[:-2] + (1, 1),
                           dtype=frame.dtype, device=bm.get_device(frame))
    for degree in range(1, s+1):
        # The first ``degree`` rows use direction 0; the last row uses direction 1.
        previous = bm.concatenate((coefficients, coefficients[..., -1:, :]), axis=-2)
        first = bm.broadcast_to(frame[..., :1, :], frame.shape[:-2] + (degree, 2))
        direction = bm.concatenate((first, frame[..., 1:, :]), axis=-2)
        zero = bm.zeros_like(previous[..., :1])
        coefficients = (
            bm.concatenate((previous, zero), axis=-1)*direction[..., 0, None]
            + bm.concatenate((zero, previous), axis=-1)*direction[..., 1, None]
        )
    multiplicity = symmetry_multiplicity(s, 2, dtype=frame.dtype)
    return coefficients / multiplicity, multiplicity
