
import numpy as np
from .Quadrature import Quadrature

class TensorProductQuadrature(Quadrature):
    def __init__(self, qfs, n=None):
        """

        Notes
        -----

        qfs 是一组积分公式
        """

        if n is None:
            n = len(qfs) # 积分公式的个数
            self.quadpts = () # 空元组
            weights = ()
            for i, qf in enumerate(qfs):
                bcs, ws = qf.get_quadrature_points_and_weights()
                self.quadpts += (bcs, )
                weights += (ws, )
        else: # n 是一个整数, qfs 是一个积分公式
            bcs, ws = qfs.get_quadrature_points_and_weights()
            self.quadpts = n*(bcs, ) 
            weights = n*(bcs, )

        # 构造 einsum 运算字符串
        s0 = 'abcdef'
        s = ''
        for i in range(n):
            s = s + s0[i]
            if i < n-1:
                s = s + ', '
        s = s + '->' + s0[:n]
        self.weights = np.einsum(s, *weights)

    def number_of_quadrature_points(self):
        n = np.product(self.weights.shape) 
        return n 
