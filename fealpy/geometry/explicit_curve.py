
import numpy as np

class LagrangeCurve:
    """
    @brief Lagrange 插值曲线
    """

    def __init__(self, node):
        self.node = node
        self.p = len(node)-1
        self.ftype = node.dtype


    def geo_dimension(self):
        return self.node.shape[-1]

    def top_dimension(self):
        return 1

    def __call__(self, xi):
        """
        @brief 
        """

        if type(xi) in {int, float}:
            bc = np.array([1-xi, xi], dtype=self.ftype)
        else:
            n = len(xi)
            bc = np.zeros((n, 2), dtype=self.ftype)
            bc[:, 0] = 1 - xi
            bc[:, 1] = xi

        TD = bc.shape[-1] - 1 
        p = self.p
        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1) # (NQ, ldof)

        ps = np.einsum('...i, id->...d', phi, self.node)
        return ps


    def multi_index_matrix(self, p):
        """
        """
        ldof = p + 1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

class BezierCurve():
    def __init__(self):
        pass


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
        
        self.dtype = node.dtype
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

        return point

    def value(self, xi):
        """
        @brief 计算一点 xi 处的样条基函数值
        """

        knot = self.knot
        node = self.node
        bval = np.zeros(self.n+1, dtype=self.dtype)

        # 初始化
        if xi == 1.0:
            bval[-2] = 1
        else:
            for i in range(self.n):
                if (xi >= knot[i]) & (xi < knot[i+1]):
                    bval[i] = 1
                    break


        for k in range(1, self.p+1):
            for i in range(self.n):
                t0 = 0 if knot[i+k] == knot[i] else (xi - knot[i])/(knot[i+k] - knot[i])
                t1 = 0 if knot[i+k+1] == knot[i+1] else (knot[i+k+1] - xi)/(knot[i+k+1] - knot[i+1]) 
                bval[i] = t0*bval[i] + t1*bval[i+1]

        point = np.einsum('j, j...->...', bval[:-1], node)
        return point


    def basis(self, xi):
        knot = self.knot
        bval = np.zeros((len(xi), self.n+1), dtype=self.dtype)


        # 初始化
        for j in range(len(xi)):
            if xi[j] == 1.0:
                bval[j, -2] = 1
            else:
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
