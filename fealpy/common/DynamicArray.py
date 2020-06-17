import numpy as np
# https://github.com/maciejkula/dynarray.git

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
                    return getattr(self.data[:self.size], name)
                return delegate
            type.__init__(cls, name, parents, attrs)
            for method_name in cls.MAGIC_METHODS:
                setattr(cls, method_name, property(make_delegate(method_name)))

    def __init__(self, data, dtype=None, capacity=1000):

        if isinstance(data, int): 
            self.shape = (data, )
            self.dtype = dtype or np.int_
            self.size = data 
            self.capacity = max(self.size, capacity)
            self.ndim = len(self.shape)
            self.data = np.empty((self.capacity,) + self._get_trailing_dimensions(),
                                  dtype=self.dtype)
        elif isinstance(data, tuple):
            self.shape = data 
            self.dtype = dtype or np.int_
            self.size = data[0] 
            self.capacity = max(self.size, capacity)
            self.ndim = len(self.shape)
            self.data = np.empty((self.capacity,) + self._get_trailing_dimensions(),
                                  dtype=self.dtype)
        elif isinstance(data, list):
            self.shape = (len(data), len(data[0])) if isinstance(data[0], list) else (len(data), )
            self.dtype = dtype or np.int_
            self.size = len(data) 
            self.capacity = max(self.size, capacity)
            self.ndim = len(self.shape)
            self.data = np.empty((self.capacity,) + self._get_trailing_dimensions(),
                                  dtype=self.dtype)
            self.data[:self.size] = data

        elif isinstance(data, np.ndarray):
            self.shape = data.shape
            self.dtype = dtype or data.dtype
            self.size = self.shape[0]
            self.capacity = max(self.size, capacity)
            self.ndim = len(self.shape)
            self.data = np.empty((self.capacity,) + self._get_trailing_dimensions(),
                                  dtype=self.dtype)
            self.data[:self.size] = data


    def _get_trailing_dimensions(self):
        return self.shape[1:]

    def __getitem__(self, idx):
        return self.data[:self.size][idx]

    def __setitem__(self, idx, value):
        self.data[:self.size][idx] = value

    def grow(self, new_size):
        self.data.resize(((new_size,) + self._get_trailing_dimensions()))
        self.capacity = new_size

    def _as_dtype(self, value):
        if hasattr(value, 'dtype') and value.dtype == self.dtype:
            return value
        else:
            return np.array(value, dtype=self.dtype)

    def append(self, value):
        """
        Append a row to the array.
        The row's shape has to match the array's trailing dimensions.
        """

        value = self._as_dtype(value)

        if value.shape != self._get_trailing_dimensions():

            value_unit_shaped = value.shape == (1,) or len(value.shape) == 0
            self_unit_shaped = self.shape == (1,) or len(self._get_trailing_dimensions()) == 0

            if value_unit_shaped and self_unit_shaped:
                pass
            else:
                raise ValueError('Input shape {} incompatible with '
                                 'array shape {}'.format(value.shape,
                                                         self._get_trailing_dimensions()))
        if self.size == self.capacity:
            self._grow(max(1, self.capacity * 2))
        self.data[self.size] = value
        self.size += 1

    def extend(self, values):
        """
        Extend the array with a set of rows.
        The rows' dimensions must match the trailing dimensions
        of the array.
        """
        values = self._as_dtype(values)

        required_size = self.size + values.shape[0]

        if required_size >= self.capacity:
            self.grow(max(self.capacity * 2,
                           required_size))

        self.data[self.size:required_size] = values
        self.size = required_size

    def shrink(self):
        """
        Reduce the array's capacity to its size.
        """
        self.grow(self.size)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return (self.data[:self.size].__repr__()
                .replace('array',
                         'DynamicArray(size={}, capacity={})'
                         .format(self._size, self._capacity)))
