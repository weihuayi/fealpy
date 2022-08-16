
import numpy as np


class BSplineCurve:
    """
    @brief B 样条曲线，它是 Bezier 样条的一般化

    一个 n 阶的样条是一个分段的 n-1 次多项式函数

    结点向量，[xi_0, xi_1, xi_2, \ldots, xi_],  元素单调非减
    * 如果互不相同，则在结点处 n-2 阶导数都连续
    * 如果 r 个结点相同，则跨过该结点处 n-r-1 阶导数连续

    basis spline

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
        assert n == len(node)
        assert n+p+1 == len(knot)
        assert np.all(knot[0:p+1] == 0)
        assert np.all(knot[-p-1:] == 1)
        
        self.ftype = node.dtype
        self.n = n 
        self.p = p
        self.knot = knot
        self.node = node

        if len(node.shape) == 1:
            self.GD = 1
        else:
            self.GD = node.shape[-1]

    def geom_dimension(self):
        return self.GD 

    def top_dimension(self):
        return 1

    def __call__(self, xi):
        """
        @brief 计算每个区间上的值
        """

        assert np.all((xi >=0) | (xi <=1))
        GD = self.GD

        if type(xi) in {int, float}:
            point = self.value(xi)
        else:
            if GD == 1:
                point = np.zeros(len(xi), dtype=self.dtype) 
            else:
                point = np.zeros((len(xi), GD), dtype=self.dtype)

            for i in range(len(xi)):
                point[i] = self.value(xi[i])

        return ps

    def value(self, xi):
        """
        @brief 计算一点 xi 处的样条基函数值
        """

        knot = self.knot
        node = self.node
        bval = np.zeros(self.n+1, dtype=self.ftype)

        # 初始化
        for i in range(self.n):
            if (xi >= self.knot[i]) & (xi < self.knot[i+1]):
                bval[i] = 1
                break

        for k in range(1, self.p):
            for i in range(self.n):
                t0 = 0 if knot[i+k] == knot[i] else (xi - knot[i])/(knot[i+k] - knot[i])
                t1 = 0 if knot[i+k+1] == knot[i+1] else (knot[i+k+1] - xi)/(knot[i+k+1] - knot[i+1]) 
                bval[i] = t0*bval[i] + t1*bval[i+1]

        point = np.einsum('j, j...->...', bval, node)
        return point


    def basis(self, xi):
        knot = self.knot
        bval = np.zeros((len(xi), self.n+1), dtype=self.ftype)


        # 初始化
        for j in range(len(xi)):
            for i in range(self.n):
                if (xi[j] >= self.knot[i]) & (xi[j] < self.knot[i+1]):
                    bval[j, i] = 1
                    break

        for j in range(len(xi)):
            for k in range(1, self.p+1):
                for i in range(self.n):
                    t0 = 0 if knot[i+k] == knot[i] else (xi[j] - knot[i])/(knot[i+k] - knot[i])
                    t1 = 0 if knot[i+k+1] == knot[i+1] else (knot[i+k+1] - xi[j])/(knot[i+k+1] - knot[i+1]) 
                    bval[j, i] = t0*bval[j, i] + t1*bval[j, i+1]

        return bval[..., :-1]






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 9
    p = 2
    knot = np.array([
        0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.8, 1, 1, 1
        ], dtype=np.float64)
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40, 60],
        [30, 90]], dtype=np.float64)
    

    curve = BSplineCurve(n, p, knot, node)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')
    
    xi = np.linspace(0, 1, 1000)
    bval = curve.basis(xi)
    fig = plt.figure()
    axes = fig.gca()
    axes.plot(xi, bval)
    plt.show()


