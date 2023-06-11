import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union


class ConformingVEMScalarSourceIntegrator2d():

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
     def source_vector(self, f):
        PI0 = self.PI0
        phi = self.smspace.basis
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        bb = self.integralalg.integral(u, celltype=True)
        g = lambda x: x[0].T@x[1]
        bb = np.concatenate(list(map(g, zip(PI0, bb))))
        gdof = self.number_of_global_dofs()
        b = np.bincount(self.dof.cell2dof, weights=bb, minlength=gdof)
        return b

