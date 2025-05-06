from ..backend import backend_manager as bm
from ..decorator import cartesian


class MembraneOscillationPDEData:
    def __init__(self, D=[0, 1, 0, 1], T=[0, 1.4]):
        """
        @brief 模型初始化函数

        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D
        self._duration = T

    def domain(self):
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self):
        """
        @brief 时间区间
        """
        return self._duration

    def solution(self, p, t):
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 返回 val
        """
        pi = bm.pi
        x, y = p[..., 0], p[..., 1]
        val = bm.cos(bm.sqrt(2)*pi*t) * bm.sin(pi*x) * bm.sin(pi*y)
        return val

    def source(self, p, t):
        """
        @brief 方程右端项 
        
        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 0
        """
        return bm.zeros_like(p[..., 0])

    def init_solution(self, p):
        """
        @brief 初值条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 返回 val
        """

        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = bm.sin(pi*x) * bm.sin(pi*y)
        return val

    def init_solution_diff_t(self, p):
        """
         @brief 初值条件的导数

         @param[in] p numpy.ndarray, 空间点
        """
        return bm.zeros_like(p[..., 0])

    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 边界条件函数值
        """
        return bm.zeros_like(p[..., 0])
