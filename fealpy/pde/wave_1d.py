from ..backend import backend_manager as bm
from ..decorator import cartesian


class StringOscillationPDEData:
    def __init__(self, D=[0, 1], T=[0, 1]):
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

    def init_solution(self, p):
        """
        @brief 初始解 u(x,0) = sin(4πx)
        @param[in] p numpy.ndarray, 空间点
        @return 初始解函数值
        """
        return bm.sin(4 * bm.pi * p)

    def init_solution_diff_t(self, p):
        """
        @brief 初始解的时间导数 u_t(x,0) = sin(8πx)
        @param[in] p numpy.ndarray, 空间点
        @return 初始解时间导数函数值
        """
        return bm.sin(8 * bm.pi * p)

    def source(self, p, t):
        """
        @brief 方程右端项 f(x,t)，此处为 0
        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点
        @return 方程右端函数值
        """
        return bm.zeros_like(p).flatten()

    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件 u(0,t)=u(1,t)=0
        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点
        @return 边界值函数值
        """
        return bm.zeros_like(p).flatten()

    def solution(self, p, t):
        """
        @brief 精确解 u(x,t) = cos(4πt)sin(4πx) + (sin(8πt)sin(8πx)) / 8π
        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点
        @return 精确解函数值
        """
        val0 = bm.cos(4*bm.pi*t) * bm.sin(4*bm.pi*p)
        val1 = (bm.sin(8*bm.pi*t) * bm.sin(8*bm.pi*p))/(8*bm.pi)
        val = val0 + val1
        return val
