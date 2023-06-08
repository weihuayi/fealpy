import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union


class ScalarSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray]):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.vector = None

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None, q=None):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        pass
