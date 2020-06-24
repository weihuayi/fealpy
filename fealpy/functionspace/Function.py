import numpy as np
from types import ModuleType


class Function(np.ndarray):
    SPACE_METHODS = ('value',
                     'grad_value',
                     'curl_value',
                     'rot_value',
                     'div_value',
                     'laplace_value',
                     'hessian_value',
                     'edge_value')

    class __metaclass__(type):
        @staticmethod
        def wrap(func):
            def outer(self, *args, **kwargs):
                val = func(self, *args, **kwargs)
                return val
            return outer

        def __new__(cls, name, parents, attrs):
            def make_delegate(name):
                def delegate(self, *args, **kwargs):
                    return getattr(self.space, name)
                return delegate
            type.__init__(cls, name, parents, attrs)
            for method_name in cls.SPACE_METHODS:
                if hasattr(self.space, method_name):
                    setattr(cls, method_name, property(make_delegate(method_name)))
                    attrs[method_name] = cls.wrap(attrs[method_name])
            return super(__metaclass__, cls).__new__(cls, name, bases, attrs)

    def __new__(cls, space, dim=None, array=None):
        if array is None:
            self = space.array(dim=dim).view(cls)
        else:
            self = array.view(cls)
        self.space = space
        return self
