import numpy as np

from scipy.sparse import csr_matrix, spdiags, eye, bmat

from typing import Optional, Union, Tuple, Callable, Any


class DirichletBC():
    def __init__(self, space: Union[Tuple, 'Space'], gD: Callable,
                 threshold: Optional[Callable] = None):
        """
        初始化 Dirichlet 边界类

        参数：
        space: 函数空间，可以是元组或者 Space 类的实例
        """
        self.space = space
        self.gD = gD
        self.threshold = threshold
        self.bctype = 'Dirichlet'

    def apply(self,
              H: np.ndarray,
              G: np.ndarray,
              f: np.ndarray,
              uh: np.ndarray = None,
              dflag: np.ndarray = None) -> Tuple[csr_matrix, np.ndarray]:
        """
        @brief 处理 Dirichlet 边界条件

        @param[in] uh: 解向量
        """
        # TODO: 注意模型是 -Delta（-） 还是 Delta（+）
        # TODO: 完善边界处理
        xi = self.space.xi
        u = self.gD(xi)
        return G, H@u - f, u

    def apply_for_other_space(self, A, f, uh) -> Tuple[csr_matrix, np.ndarray]:
        raise NotImplemented

    def apply_for_vspace_with_scalar_basis(self, A, f, uh, dflag=None):
        raise NotImplemented
