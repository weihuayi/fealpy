from ..backend import backend_manager as bm
from ..decorator import cartesian

class StringOscillationPDEData: 
    def __init__(self, D = [0, 1], T = [0, 4]):
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
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        val = bm.zeros_like(p)
        flag = p < 0.7
        val[flag] = 0.5/7.0*p[flag] 
        val[~flag] = 0.5/3.0*(1 - p[~flag])
        return val 

    def init_solution_diff_t(self, p):
        """
        @brief 初始解的时间导数函数(初始速度条件)

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解时间导数函数值
        """
        return bm.zeros_like(p)

    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点

        @return 方程右端函数值
        """
        return bm.zeros_like(p)
    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点

        """
        return bm.zeros_like(p) 
