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
                    return getattr(self._data[:self._size], name)
                return delegate
            type.__init__(cls, name, parents, attrs)
            for method_name in cls.MAGIC_METHODS:
                setattr(cls, method_name, property(make_delegate(method_name)))

    def __init__(self, array_or_shape=(None,), dtype=None, capacity=10,
                 allow_views_on_resize=False):

        if isinstance(array_or_shape, tuple):
            if not len(array_or_shape) or array_or_shape[0] is not None:
                raise ValueError('The shape argument must be a non-empty tuple '
                                 'and have None as the first dimension')
            self._shape = array_or_shape
            self._dtype = dtype
            self._size = 0
            self._capacity = capacity
        elif isinstance(array_or_shape, np.ndarray):
            self._shape = (None,) + array_or_shape.shape[1:]
            self._dtype = dtype or array_or_shape.dtype
            self._size = array_or_shape.shape[0]
            self._capacity = max(self._size, capacity)

        self._data = np.empty((self._capacity,) + self._get_trailing_dimensions(),
                              dtype=self._dtype)

        if isinstance(array_or_shape, np.ndarray):
            self[:] = array_or_shape
        self._allow_views_on_resize = allow_views_on_resize

    def _get_trailing_dimensions(self):
        return self._shape[1:]

    def __getitem__(self, idx):
        return self._data[:self._size][idx]

    def __setitem__(self, idx, value):
        self._data[:self._size][idx] = value

    def _grow(self, new_size):

        try:
            self._data.resize(((new_size,) + self._get_trailing_dimensions()))
        except ValueError as e:
            if 'an array that references' in e.message:
                if self._allow_views_on_resize:
                    self._data = np.resize(self._data,
                                           ((new_size,) + self._get_trailing_dimensions()))
                else:
                    raise ValueError('Unable to grow the array '
                                     'as it refrences or is referenced '
                                     'by another array. Growing the array '
                                     'would result in views pointing at stale data. '
                                     'You can suppress this exception by setting '
                                     '`allow_views_on_resize=True` when instantiating '
                                     'a DynamicArray.')
            else:
                raise

        self._capacity = new_size

    def _as_dtype(self, value):
        if hasattr(value, 'dtype') and value.dtype == self._dtype:
            return value
        else:
            return np.array(value, dtype=self._dtype)

    def append(self, value):
        """
        Append a row to the array.
        The row's shape has to match the array's trailing dimensions.
        """

        value = self._as_dtype(value)

        if value.shape != self._get_trailing_dimensions():

            value_unit_shaped = value.shape == (1,) or len(value.shape) == 0
            self_unit_shaped = self._shape == (1,) or len(self._get_trailing_dimensions()) == 0

            if value_unit_shaped and self_unit_shaped:
                pass
            else:
                raise ValueError('Input shape {} incompatible with '
                                 'array shape {}'.format(value.shape,
                                                         self._get_trailing_dimensions()))
        if self._size == self._capacity:
            self._grow(max(1, self._capacity * 2))
        self._data[self._size] = value
        self._size += 1

    def extend(self, values):
        """
        Extend the array with a set of rows.
        The rows' dimensions must match the trailing dimensions
        of the array.
        """
        values = self._as_dtype(values)

        required_size = self._size + values.shape[0]

        if required_size >= self._capacity:
            self._grow(max(self._capacity * 2,
                           required_size))

        self._data[self._size:required_size] = values
        self._size = required_size

    def shrink(self):
        """
        Reduce the array's capacity to its size.
        """
        self._grow(self._size)

    @property
    def shape(self):
        return (self._size,) + self._get_trailing_dimensions()

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return (self._data[:self._size].__repr__()
                .replace('array',
                         'DynamicArray(size={}, capacity={})'
                         .format(self._size, self._capacity)))
