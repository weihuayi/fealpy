import numpy as np

class CRSplineCurve:
    """
    @brief Catmull-Rom Spline Curve 

    @note 该算法生成的曲线满足如下性质
    1. p_i 点的切线方向和 p_{i+1} - p_{i-1} 平行
    2. 穿过所有的控制点
    
    在每相邻两点之间，取四个条件：过两点，两点的切线

    三次 Hermite 样条
    """
    def __init__(self, node, tau=0.2):
        """
        @param[in] node  控制点数组
        """

        self.ftype = node.dtype
        NN = len(node) 

        assert NN >= 2

        GD = node.shape[-1]
        self.node = np.zeros((NN+2, GD), dtype=self.ftype)
        self.node[1:-1] = node
        self.node[0] = 2*node[0] - node[1]
        self.node[-1] = 2*node[-1] - node[-2]
        self.tau = tau

        self.M = np.array([
            [0, 1, 0, 0],
            [-tau, 0, tau, 0],
            [2*tau, tau-3, 3-2*tau, -tau],
            [-tau, 2-tau, tau-2, tau]], dtype=self.ftype)

    def geom_dimension(self):
        return self.node.shape[-1]

    def top_dimension(self):
        return 1

    def __call__(self, xi):
        """
        @brief 计算每个区间上的值
        """
        if type(xi) in {int, float}:
            bc = np.array([1, xi, xi, xi], dtype=self.ftype)
        else:
            n = len(xi)
            bc = np.zeros((n, 4), dtype=self.ftype)
            bc[:, 0] = 1
            bc[:, 1:] = xi[:, None]

        np.cumprod(bc[..., 1:], axis=-1, out=bc[..., 1:])
        M = bc@self.M

        NN = len(self.node) - 2
        index = np.zeros((NN-1, 4), dtype=np.int_)
        index[:, 0] = range(0, NN-1)
        index[:, 1] = index[:, 0] + 1
        index[:, 2] = index[:, 1] + 1
        index[:, 3] = index[:, 2] + 1

        GD = self.geom_dimension() 
        # node[index].shape == (NN-1, 4, GD)
        # M.shape == (NQ, 4)
        ps = np.einsum('...j, kjd->k...d', M, self.node[index]).reshape(-1, GD)

        return ps



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40,60]], dtype=np.float64)


    c0 = CRSplineCurve(node, 0.2)
    c1 = CRSplineCurve(node, 0.5)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')

    xi = np.linspace(0, 1, 1000)
    ps0 = c0(xi)
    ps1 = c1(xi)
    axes.plot(ps0[:, 0], ps1[:, 1], 'r', ps1[:, 0], ps1[:, 1], 'k')
    plt.show()


