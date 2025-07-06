from fealpy.backend import backend_manager as bm
from ..decorator  import cartesian 
from ..mesh import TriangleMesh
import sympy as sp


class BoxDomainData():
    """
    @brief 混合边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """
    #def __init__(self, u1, u2, E, nu):
    def __init__(self, u1, u2, mu, lam):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        #self.E = E 
        #self.nu = nu

        #self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        #self.mu = self.E/(2*(1+self.nu))
        self.lam = lam
        self.mu = mu
        #mu = self.mu
        #lam = self.lam
        x,y = sp.symbols('x, y')

        u = sp.Matrix([u1, u2])

        grad_u = sp.Matrix([[sp.diff(u1, x), sp.diff(u1, y)],
                    [sp.diff(u2, x), sp.diff(u2, y)]])
        u1x = sp.diff(u1, x)
        u1y = sp.diff(u1, y)
        u2x = sp.diff(u2, x)
        u2y = sp.diff(u2, y)
        #grad_u1 = sp.Matrix([sp.diff(u1, x), sp.diff(u1, y)])
        #grad_u2 = sp.Matrix([sp.diff(u2, x), sp.diff(u2, y)])

        # 对称梯度（应变张量）ε(u) = 0.5*(∇u + ∇u.T)
        eps = 0.5 * (grad_u + grad_u.T)
        print("eps:", eps)

        # 散度 div u
        div_u = sp.diff(u1, x) + sp.diff(u2, y)

        # 单位张量 I
        I = sp.eye(2)

        # 应力张量 σ(u) = 2μ ε(u) + λ (div u) I
        sigma = 2 * mu * eps + lam * div_u * I
        div_sigma = -sp.Matrix([
            sp.diff(sigma[0, 0], x) + sp.diff(sigma[0, 1], y),
            sp.diff(sigma[1, 0], x) + sp.diff(sigma[1, 1], y)
        ])
        print("div_sigma:", div_sigma)
        self.fx = sp.lambdify((x, y), div_sigma[0], "numpy")
        self.fy = sp.lambdify((x, y), div_sigma[1], "numpy")
        self.u1 = sp.lambdify((x, y), u1, "numpy")
        self.u2 = sp.lambdify((x, y), u2, "numpy")
        self.u1x = sp.lambdify((x, y), u1x, "numpy")
        self.u1y = sp.lambdify((x, y), u1y, "numpy")
        self.u2x = sp.lambdify((x, y), u2x, "numpy")
        self.u2y = sp.lambdify((x, y), u2y, "numpy")


        #self.grad_u1 = sp.lambdify((x, y), grad_u1, "numpy")
        #self.grad_u2 = sp.lambdify((x, y), grad_u2, "numpy")
        #self.gradu = sp.lambdify((x, y), grad_u, "numpy")


    def domain(self):
        return [0, 1, 0, 1]

    def init_mesh(self, n = 1):
        """
        @brief 初始化网格
        @param[in] n 网格加密的次数，默认值为 1
        @return 返回一个初始化后的网格对象
        """
        h = 0.5
        domain = RectangleDomain()
        mesh = TriangleMesh.from_domain_distmesh(domain, h, output=False)
        mesh.uniform_refine(n)

        return mesh 

    def triangle_mesh(self):
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2)

        return mesh

    @cartesian
    def source(self, p):
        """
        @brief 返回给定点的源项值 f
        @param[in] p 一个表示空间点坐标的数组
        @return 返回源项值
        """
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.fx(x, y)
        val[..., 1] = self.fy(x, y)
        #val[..., 0] = 2.0*self.mu*bm.sin(x) + self.lam*bm.sin(x)
        #val[..., 1] = 2.0*self.mu*bm.sin(y) + self.lam*bm.sin(y)
        #val[..., 0] = self.lam *bm.sin(x)*bm.sin(y) + 3*self.mu*bm.sin(x)*bm.sin(y)
        #val[..., 1] =-self.lam*(-bm.sin(y)+bm.cos(x)*bm.cos(y))+2*self.mu*bm.sin(y)-self.mu*bm.cos(x)*bm.cos(y)

        #val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        #val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)
        #val[..., 0] = -2*self.lam-4*self.mu
        #val[..., 1] = 0
        #val[..., 0] = 0
        #val[..., 1] = -self.lam -self.mu
        #val[..., 0] = 0
        #val[..., 1] = -2*self.mu
        #pi = bm.pi
        #val[..., 0] = 22.5*bm.pi**2/13*bm.sin(pi*x)*bm.sin(pi*y)
        #val[..., 1] = -12.5*pi**2/13*bm.cos(pi*x)*bm.cos(pi*y)
        return val
    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.u1(x, y)
        val[..., 1] = self.u2(x, y)
        #val[..., 0] = bm.sin(x)*bm.sin(y)
        #val[..., 0] = bm.sin(x)
        #val[..., 1] = bm.sin(y)
        #val[..., 0] = x**2
        #val[..., 1] = 0
        #val[..., 0] = x*(1-x)*y*(1-y)
        #val[..., 1] = 0
        #val[..., 0] = 0
        #val[..., 1] = x**2
        #pi = bm.pi
        #val[..., 0] = bm.sin(pi*x)*bm.sin(pi*y)
        #val[..., 1] = 0
        return val
    @cartesian
    def solution1(self, p):
        x = p[..., 0]
        y = p[..., 1]

        #val = bm.sin(x)
        #val = bm.sin(x)*bm.sin(y)

        val = self.u1(x, y)
        #val = x**2
        #val = x*(1-x)*y*(1-y)
        #val = 0*x
        #pi = bm.pi
        #val = bm.sin(pi*x)*bm.sin(pi*y)

        return val
    @cartesian
    def solution2(self, p):
        x = p[..., 0]
        y = p[..., 1]

        #val= bm.sin(y)
        val = self.u2(x, y)
        #val= bm.ones(x.shape, dtype=bm.float64)
        #val = 0*x
        #val = x**2
        return val
    @cartesian
    def gradient1(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.u1x(x, y)
        val[..., 1] = self.u1y(x, y)
        #val = bm.sin(x)
        #val = bm.sin(x)*bm.sin(y)
        #val = x**2
        #val = x*(1-x)*y*(1-y)
        #val = 0*x
        #pi = bm.pi
        #val = bm.sin(pi*x)*bm.sin(pi*y)

        return val
    @cartesian
    def gradient2(self, p):
        x = p[..., 0]
        y = p[..., 1]

        #val= bm.sin(y)
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.u2x(x, y)
        val[..., 1] = self.u2y(x, y)
        #val[..., 0] = self.grad_u2(x, y)[0]
        #val[..., 1] = self.grad_u2(x, y)[1]

        #val = self.grad_u2(x, y)
        #val= bm.ones(x.shape, dtype=bm.float64)
        #val = 0*x
        #val = x**2
        return val


    @cartesian
    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]

        #val= bm.sin(y)
        val = self.u2(x, y)
        return


    @cartesian
    def dirichlet(self, p):
        """
        @brief 返回 Dirichlet 边界上的给定点的位移
        @param[in] p 一个表示空间点坐标的数组
        @return 返回位移值，这里返回常数向量 [0.0, 0.0]
        """
        #val = bm.zeros((p.shape[0], 2), dtype=bm.float64)
        # val = bm.array([0.0, 0.0], dtype=bm.float64)


        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定点是否在 Dirichlet 边界上
        @param[in] p 一个表示空间点坐标的数组
        @return 如果在 Dirichlet 边界上，返回 True，否则返回 False
        """
        x = p[..., 0]
        y = p[..., 1]
        flag1 = bm.abs(x) < 1e-13
        flag2 = bm.abs(x - 1) < 1e-13
        flagx = bm.logical_or(flag1, flag2)
        flag3 = bm.abs(y) < 1e-13
        flag4 = bm.abs(y - 1) < 1e-13
        flagy = bm.logical_or(flag3, flag4)
        flag = bm.logical_or(flagx, flagy)

        return flag



