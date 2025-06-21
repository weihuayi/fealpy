
import jax.numpy as jnp

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
            A: csr_matrix, 
            f: jnp.ndarray, 
            uh: jnp.ndarray=None, 
            dflag: jnp.ndarray=None) -> Tuple[csr_matrix, jnp.ndarray]:
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


    def apply_for_other_space(self, A, f, uh) -> Tuple[csr_matrix, jnp.ndarray]:
        """
        @brief 处理基是向量函数的向量函数空间或标量函数空间的 Dirichlet 边界条件
        """
        space = self.space
        gD = self.gD
        uh, isDDof = space.boundary_interpolate(gD, uh, threshold=self.threshold) # isDDof.shape == uh.shape
        f = f - A@uh.reshape(-1) # 注意这里不修改外界 f 的值

        bdIdx = jnp.zeros(A.shape[0], dtype=jnp.int_)
        bdIdx = bdIdx.at[isDDof.reshape(-1)].set(1)
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1

        f = f.at[isDDof.reshape(-1)].set(uh[isDDof].reshape(-1))
        return A, f 

    def apply_for_vspace_with_scalar_basis(self, A, f, uh, dflag=None):
        """
        @brief 处理基由标量函数组合而成的向量函数空间的 Dirichlet 边界条件

        @param[in] 

        """
        space = self.space
        assert isinstance(space, tuple) and not isinstance(space[0], tuple)

        gD = self.gD
        if dflag is None:
            dflag = space[0].boundary_interpolate(gD, uh, threshold=self.threshold)
        f = f - A@uh.flat # 注意这里不修改外界 f 的值

        bdIdx = jnp.zeros(A.shape[0], dtype=jnp.int_)
        bdIdx[dflag.flat] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        f[dflag.flat] = uh.ravel()[dflag.flat]
        return A, f 
