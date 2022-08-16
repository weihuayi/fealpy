
import numpy as np


class BSplineCurve:
    """
    @brief B 样条曲线，它是 Bezier 样条的一般化

    一个 n 阶的样条是一个分段的 n-1 次多项式函数

    结点向量，[t_0, t_1, t_2, \ldots, t_n],  元素单调非减
    * 如果互不相同，则在结点处 n-2 阶导数都连续
    * 如果 r 个结点相同，则跨过该结点处 n-r-1 阶导数连续

    basis spline

    The open knot vectors, the first and last knots appear p+1 

    m = n + p + 1
    """
    def __init__(self, p, knot, node):
        """
        @param[in] p
        @param[in] knot  结向量 
        @param[in] node  控制点数组


        @note 
            knot[0:p+1] == 0
            knot[-p-1:] == 1
        """
        self.p = p
        self.knot = knot
        self.node = node




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = 3
    knot = np.array([
        0, 0, 0, 0, 0.25, 0.4, 0.6, 0.75, 1, 1, 1, 1 
        ], dtype=np.float64)
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40,60]], dtype=np.float64)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')
    plt.show()


