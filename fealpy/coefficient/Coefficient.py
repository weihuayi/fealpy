import numpy as np


class Coefficient:
    """
    @brief 
    """
    def __init__(self, t=0.0):
        self.time = t



class ConstantCoefficient(Coefficient):
    """
    @brief 不依赖于时间和空间的常数
    """

    def __init__(self, c=1.0):
        self.constant = c
