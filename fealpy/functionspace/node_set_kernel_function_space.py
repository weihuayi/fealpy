import numpy as np

class NodeSetKernelSpace:

    def __init__(self, mesh, ker):
        """
        @brief 

        @param[in] mesh NodeSet 类型的网格
        @param[in] ker str kernel 函数计算的字符串
        """
        self.mesh = mesh

    def kernel(self, r):
        pass

    def grad_kernel(self, r):
        pass

    def value(self, u, points):
        """
        @brief 
        @param[in] u 定义在节点上的物理量的值
        @param[in] points 需要求值的点 

        @return 物理量在 points 处的函数值
        """
        pass


    def grad_value(self, u, points):
        """
        @brief 
        @param[in] u 定义在节点上的物理量的值
        @param[in] points 需要求值的点 

        @return 物理量在 points 处的梯度值
        """
        pass

