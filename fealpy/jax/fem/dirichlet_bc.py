
from jax.experimental.sparse import BCOO
import jax.numpy as jnp

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
            A: BCOO, 
            f: jnp.ndarray, 
            uh: jnp.ndarray=None, 
            dflag: jnp.ndarray=None) -> Tuple[BCOO, jnp.ndarray]:
        """
        @brief 处理 Dirichlet 边界条件  

        @param[in] A: 系数矩阵
        @param[in] f: 右端向量
        @param[in] uh: 解向量
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            gdof = self.space[0].number_of_global_dofs()
            GD = int(A.shape[0]//gdof)
            if uh is None:
                uh = self.space[0].function(dim=GD)

            return self.apply_for_vspace_with_scalar_basis(A, f, uh, dflag=dflag)
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            gdof = self.space.number_of_global_dofs()
            GD = int(A.shape[0]//gdof)
            if uh is None:
                uh = self.space.function(dim=GD)  

            return self.apply_for_other_space(A, f, uh)


    