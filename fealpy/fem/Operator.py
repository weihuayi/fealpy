import numpy as np


class Operator():
    """
    @brief 表示一个一般的算子 y = A(x)

    其中 x 是输入， y 是输出
    """
    def __init__(self, shape=(0, 0)):
        self.shape = shape 

    
    def rows(self):
        """
        @brief 获取算子的输出的长度
        """
        return self.shape[0]


    def cols(self):
        """
        @brief 获取算子的输入的长度
        """
        return self.shape[1]


    def mult(self, x):
        """
        @brief 计算 y = A(x), 具体由子类实现
        """
        raise NotImplementedError

    def transpose_mult(self, x):
        """
        @brief 计算 y = A^T(x), 具体由子类实现
        """
        raise NotImplementedError
