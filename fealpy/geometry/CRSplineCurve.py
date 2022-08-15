import numpy as np

class CRSplineCurve:
    """
    @brief Catmull-Rom Spline Curve 
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


