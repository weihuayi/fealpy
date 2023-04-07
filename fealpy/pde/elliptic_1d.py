import numpy as np
from ..decorator import cartesian

class SinPDEData:
    def __init__(self, D=[0, 1]):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        """
        self._domain = D 

    def domain(self):
        """
        @brief 空间区间
        """
        return self._domain
