
from typing import Any, Tuple, List, Dict, Optional, Literal, Callable, TypeVar, Generic
from ..backend import TensorLike as _DT
from ..typing import Scalar

_RT = TypeVar('_RT', bound=Scalar)


class Objective(Generic[_RT]):
    """Objective function for optimizers.
    Objective objects can be constructed by directly passing in functions or by inheritance.

    Example:
    Directly passing in:

        obj = Objective(my_fun, my_jac, ...)

    Using decorators:

        @Objective
        def obj(x):
            return sum(x**2 + 2*x)

        @obj.set_jac
        def my_jac(x):
            return 2*x + 2

    Inheritance:

        class MyObjective(Objective):
            def fun(x):
                # some code

            def jac(x):
                # some code

        obj = MyObjective('may have some args')
    """
    args = tuple()
    kwargs = dict()

    @classmethod
    def partial(cls, *args, **kwargs):
        """Build objective function with extra args."""
        def _partial_objective(fun: Callable, /):
            obj = cls(fun)
            obj.args = args
            obj.kwargs = kwargs
            return obj
        return _partial_objective

    def __init__(self, fun: Optional[Callable[..., _RT]] = None, /,
                 jac: Optional[Callable[..., _DT]] = None,
                 hess: Optional[Callable[..., _DT]] = None,
                 hessp: Optional[Callable[..., _DT]] = None,
                 *,
                 bounds: Optional[Tuple[_DT, _DT]] = None,
                 constraints: List['Constraint'] = [],
                 args: Optional[Tuple] = None,
                 kwargs: Optional[Dict] = None) -> None:
        """Build an objective for optimizers.

        Parameters:
            fun (Callable | None, optional): The objective function to be minimized.
            jac (Callable | None, optional): Method for computing the gradient vector.
            hess (Callable | None, optional): Method for computing the Hessian matrix.
            hessp (Callable | None, optional): Hessian of objective function times an arbitrary vector p.
            bounds (Tuple[Tensor, Tensor] | None, optional):
            constraints (List[Constraint], optional):
            args (Tuple | None, optional): _description_. Defaults to None.
            kwargs (Dict | None, optional): _description_. Defaults to None.
        """
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp
        self.bounds = bounds
        self.constraints = constraints

        if args is not None:
            self.args = tuple(args)

        if kwargs is not None:
            self.kwargs = dict(kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> _RT:
        return self.fun(*args, **kwds)

    def set_jac(self, fun: Callable[..., _DT], /):
        self._jac = fun
        return fun

    def set_hess(self, fun: Callable[..., _DT], /):
        self._hess = fun
        return fun

    def set_hessp(self, fun: Callable[..., _RT], /):
        self._hessp = fun
        return fun

    def set_bounds(self, minimum: _DT, maximum: _DT, /) -> None:
        self.bounds = minimum, maximum

    def add_constraint(self, cons: 'Constraint', /) -> 'Constraint':
        self.constraints.append(cons)
        return cons

    def fun(self, x: _DT) -> _RT:
        if self._fun is None:
            raise NotImplementedError
        return self._fun(x, *self.args, **self.kwargs)

    def jac(self, x: _DT) -> _DT:
        if self._jac is None:
            raise NotImplementedError
        return self._jac(x, *self.args, **self.kwargs)

    def hess(self, x: _DT) -> _DT:
        if self._hess is None:
            raise NotImplementedError
        return self._hess(x, *self.args, **self.kwargs)

    def hessp(self, x: _DT, p: _DT) -> _RT:
        if self._hessp is None:
            raise NotImplementedError
        return self._hessp(x, p, *self.args, **self.kwargs)


class Constraint():
    args = tuple()
    kwargs = dict()

    def __init__(self, fun: Callable[..., float], /,
                 jac: Optional[Callable[..., _DT]] = None,
                 *,
                 type: Literal['ineq', 'eq'] = 'ineq',
                 args: Optional[Tuple] = None,
                 kwargs: Optional[Dict] = None) -> None:
        self._fun = fun
        self._jac = jac
        self.type = type

        if args is not None:
            self.args = args

        if kwargs is not None:
            self.kwargs = kwargs

    def set_jac(self, fun: Callable[..., _DT], /) -> None:
        self._jac = fun

    def fun(self, x: _DT) -> float:
        return self._fun(x)

    def jac(self, x: _DT) -> _DT:
        if self._jac is None:
            raise NotImplementedError
        return self._jac(x)
