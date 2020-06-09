import numpy as np


class DynamicArray(object):
    MAGIC_METHODS = ('__radd__',
                     '__add__',
                     '__sub__',
                     '__rsub__',
                     '__mul__',
                     '__rmul__',
                     '__div__',
                     '__rdiv__',
                     '__pow__',
                     '__rpow__',
                     '__eq__',
                     '__len__')

    class __metaclass__(type):
        def __init__(cls, name, parents, attrs):

            def make_delegate(name):

                def delegate(self, *args, **kwargs):
                    return getattr(self._data[:self._size], name)

                return delegate

            type.__init__(cls, name, parents, attrs)

            for method_name in cls.MAGIC_METHODS:
                setattr(cls, method_name, property(make_delegate(method_name)))
