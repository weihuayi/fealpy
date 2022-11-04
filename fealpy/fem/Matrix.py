
import numpy as np

class Matrix(Operator):
    """
    """

    def __init__(self, shape):
        super().__init__(shape)

    def is_square(self):
        return self.shape[0] == self.shape[1]



