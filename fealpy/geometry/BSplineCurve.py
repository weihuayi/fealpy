
import numpy as np


class BSplineCurve:
    """

    The open knot vectors, the first and last knots appear p+1 

    m = n + p + 1
    """
    def __init__(self, n, p, knot, node):
        """
        @param[in] n 控制点的个数
        @param[in] p 样条基函数的次数
        @param[in] knot  结向量 
        @param[in] node  控制点数组


        @note 
            knot[0:p+1] == 0
            knot[-p-1:] == 1
        """
        assert n+p+1 == len(knot)
        assert np.all(knot[0:p+1] == 0)
        assert np.all(knot(-p-1:] == 1)

        self.n = n 
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


