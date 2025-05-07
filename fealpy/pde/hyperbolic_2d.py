from ..backend import backend_manager as bm
from ..decorator import cartesian

class Hyperbolic2dPDEData:
    def domain(self):
        """
        @brief 空间定义域
        @return [(float, float), (float, float)]
        """
        return [(0.0, 2.0), (0.0, 2.0)]

    def duration(self):
        """
        @brief 时间定义域
        """
        return [0.0, 4.0]

    def a(self):
        """
        @brief 方程中常数 a
        """
        return 1.0

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始条件 u(x, y, 0) = |xy - 1|
        """
        x = p[..., 0]
        y = p[..., 1]
        return bm.abs(x * y - 1)

    @cartesian
    def solution(self, p, t):
        """
        @brief 真解函数：
        u(x,y,t) = 分段定义关于 xy 的函数
        """
        x = p[..., 0]
        y = p[..., 1]
        xy = x * y

        val = bm.zeros_like(xy)

        flag1 = xy <= t
        flag2 = xy > t + 1
        flag3 = (~flag1) & (~flag2)

        val = bm.where(flag1, 1.0, val)
        val = bm.where(flag3, 1.0 - (xy - t), val)
        val = bm.where(flag2, (xy - t) - 1.0, val)

        return val

    @cartesian
    def source(self, p, t):
        """
        @brief 源项函数，恒为 0
        """
        return bm.zeros(p.shape[0])

    @cartesian
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件，在特定点为 1，其余为 0
        """
        x = p[..., 0]
        y = p[..., 1]
        tol = 1e-12

        cond = ((bm.abs(x - 0.0) < tol) & (bm.abs(y - 0.0) < tol)) | \
            ((bm.abs(x - 0.0) < tol) & (bm.abs(y - 1.0) < tol)) | \
            ((bm.abs(x - 1.0) < tol) & (bm.abs(y - 0.0) < tol))

        val = bm.zeros_like(x)
        val = bm.where(cond, 1.0, val)
        return val

