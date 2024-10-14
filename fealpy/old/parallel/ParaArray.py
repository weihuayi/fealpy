import numpy as np

class ParaArray(np.ndarray):
    """
    并行对象
    """
    def __new__(cls, lidx, shape=None,  array=None, dtype=np.float):
        if array is None:
            if shape is None:
                raise ValueError("`shape` and `array` can not simultaneously be None ")
            self = np.zeros(shape, dtype=dtype).view(cls)
        else:
            self = array.view(cls)

        self.lidx = lidx
        return self

    def local_update(self, data):
        self[self.lidx] = data

