
import inspect

import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, Callable, Tuple, Union

class ConformingVEMScalarSourceIntegrator2d():
    def __init__(self, f: Union[Callable, int, float, NDArray], PI0, c=None):
        """
        @brief

        @param[in] f 
        """
        self.coef = 1 if c is None else c
        self.f = f
        self.vector = None
        self.PI0 = PI0

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None, q=None):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        f = self.f
        N = len(inspect.signature(f).parameters)
        phi = space.smspace.basis
        p = space.p
        q = p + 3 if q is None else q
        if N==1:
            def u(x, index):
                return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        else:
            def u(x, index):
                return np.einsum('ij, ijm->ijm', f(x, index), phi(x, index=index))
        bb = space.mesh.integral(u, q=q, celltype=True)
        if isinstance(self.coef, (int, float)):
            g = lambda x: x[0].T@x[1]
            bb = np.concatenate(list(map(g, zip(self.PI0, bb))))
            return self.coef*bb
        elif isinstance(self.coef, np.ndarray):
            g = lambda x: (x[0].T@x[1])*x[2]
            bb = np.concatenate(list(map(g, zip(self.PI0, bb, self.coef))))
        return bb

class NonConformingVEMScalarSourceIntegrator2d():
    def __init__(self, f: Union[Callable, int, float, NDArray], PI0):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.vector = None
        self.PI0 = PI0

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None, q=None):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        f = self.f
        phi = space.smspace.basis
        p = space.p
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        bb = space.mesh.integral(u, q=p+3, celltype=True)
        g = lambda x: x[0].T@x[1]
        bb = np.concatenate(list(map(g, zip(self.PI0, bb))))
        return bb
