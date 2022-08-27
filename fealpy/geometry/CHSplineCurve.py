
import numpy as np

class CHSplineCurve:
    """
    @brief 三次 Hermite 样条曲线 

    @note 注意三次 Hermite 样条曲线比 Catmull-Rom 样条更一般


    h_{00} = 2t^3 - 3t^2 + 1 = (1 + 2t)(1-t)^2 = B_0(t) + B_1(t)
    h_{01} = -2t^3 + 3t^2    = t^2(3 - 2t)     = B_1(t)/3
    h_{10} = t^3 - 2t^2 + t  = t(1 - t)^2      = B_3(t) + B_2(t)
    h_{11} = t^3 - t^2       = t^2(t - 1)      = -B_2(t)/3
    p(t) = h_{00} p_0 + h_{01} p_1 + h_{10} m_0 + h_{11} m_1


    p(t) = p_0 + m_0 t + (-3p_0 + 3p_1 - 2m_0 - m_1) t^2 + (2p_0 - 2p_1 + m_0 + m_1)

    """
    def __init__(self, node, tang):
        """
        @param[in] node  控制点数组
        @param[in] tangent 切线方向
        """

        self.ftype = node.dtype
        NN = len(node) 

        assert NN >= 2

        GD = node.shape[-1]
        self.node = node
        self.tang = tang

        self.M = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [-3, 3, -2, -1],
            [2, -2, 1, 1]], dtype=self.ftype)

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

        NN = len(self.node)

        GD = self.geom_dimension() 
        # node[index].shape == (NN-1, 4, GD)
        # M.shape == (NQ, 4)

        cnode = np.zeros((NN-1, 4, GD), dtype=self.ftype)
        cnode[:, 0, :] = self.node[0:-1]
        cnode[:, 1, :] = self.node[1:]
        cnode[:, 2, :] = self.tang[0:-1]
        cnode[:, 3, :] = self.tang[1:]
        
        ps = np.einsum('...j, kjd->k...d', M, cnode).reshape(-1, GD)

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


    tang0 = np.zeros_like(node)
    tang0[1:-1] = node[2:] - node[0:-2]
    tang0[0] = 2*node[1] - 2*node[0]
    tang0[-1] = 2*node[-1] - 2*node[-2]
    c0 = CHSplineCurve(node, 0.2*tang0)
    c1 = CHSplineCurve(node, 0.5*tang0)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')

    xi = np.linspace(0, 1, 1000)
    ps0 = c0(xi)
    ps1 = c1(xi)
    axes.plot(ps0[:, 0], ps1[:, 1], 'r', ps1[:, 0], ps1[:, 1], 'k')
    plt.show()


