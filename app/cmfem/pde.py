"""Manufactured Poisson and polyharmonic problems for the examples."""

from functools import lru_cache

import sympy as sp
from fealpy.backend import backend_manager as bm


class PolyharmonicPDE:
    """Manufactured solution of ``(-1)**p Delta**p u = f``.

    Here ``m`` is the smoothness order and ``p=m+1``.  Derivative callables
    use FEALPy's independent symmetric-component ordering.
    """

    def __init__(self, expression, dimension, m):
        self.expression = expression
        self.dimension = dimension
        self.m = m
        self.order = m + 1
        self.symbols = sp.symbols('x y z')[:dimension]
        operator = expression
        for _ in range(self.order):
            terms = [sp.diff(operator, variable, 2)
                     for variable in self.symbols]
            operator = -sum(terms)
        self._solution = sp.lambdify(self.symbols, expression, 'numpy')
        self._source = sp.lambdify(self.symbols, operator, 'numpy')

    def _evaluate(self, function, point):
        coordinate = tuple(point[..., i] for i in range(self.dimension))
        value = bm.asarray(function(*coordinate))
        return value + bm.zeros(point.shape[:-1], dtype=point.dtype)

    def solution(self, point):
        return self._evaluate(self._solution, point)

    def dirichlet(self, point):
        return self.solution(point)

    def source(self, point):
        return self._evaluate(self._source, point)

    @lru_cache(maxsize=None)
    def derivatives(self, order):
        """Return Cartesian symmetric derivative functions through ``order``."""
        derivatives = []
        for degree in range(order + 1):
            indices = bm.multi_index_matrix(degree, self.dimension-1,
                                            dtype=bm.int32)
            indices = bm.tolist(indices)
            expressions = []
            for alpha in indices:
                derivative = self.expression
                for variable, count in zip(self.symbols, alpha):
                    derivative = sp.diff(derivative, variable, int(count))
                expressions.append(derivative)

            function = sp.lambdify(self.symbols, expressions, 'numpy', cse=True)

            def evaluate(point, function=function, degree=degree):
                coordinate = tuple(point[..., i] for i in range(self.dimension))
                shape = point.shape[:-1]
                values = [bm.broadcast_to(bm.asarray(value), shape)
                          for value in function(*coordinate)]
                return values[0] if degree == 0 else bm.stack(values, axis=-1)

            derivatives.append(evaluate)
        return tuple(derivatives)

    def gradient(self, point):
        return self.derivatives(1)[1](point)

    def hessian(self, point):
        return self.derivatives(2)[2](point)

    def grad_3(self, point):
        return self.derivatives(3)[3](point)


class ManufacturedPDE(PolyharmonicPDE):
    """A manufactured function used for interpolation tests."""

    def __init__(self, expression, dimension):
        self.expression = expression
        self.dimension = dimension
        self.m = None
        self.order = None
        self.symbols = sp.symbols('x y z')[:dimension]
        self._solution = sp.lambdify(self.symbols, expression, 'numpy')

    def source(self, point):
        raise NotImplementedError("Interpolation data have no source term.")


class LaplacePDE(PolyharmonicPDE):
    def __init__(self, expression, dimension=2):
        super().__init__(expression, dimension, m=0)


class DoubleLaplacePDE(PolyharmonicPDE):
    def __init__(self, expression, dimension=2):
        super().__init__(expression, dimension, m=1)


class TripleLaplacePDE(PolyharmonicPDE):
    def __init__(self, expression, dimension=2):
        super().__init__(expression, dimension, m=2)
