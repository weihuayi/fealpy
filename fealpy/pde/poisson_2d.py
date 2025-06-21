
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian, barycentric

class CosCosData:
    """
        -\\Delta u = f
        u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self, kappa=1.0):
        self.kappa = kappa # Robin 条件中的系数

    def domain(self):
        """
        @brief 模型定义域
        """
        return [0, 1, 0, 1]

    @cartesian
    def solution(self, p):
        """  
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.cos(pi*x)*bm.cos(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 2*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)
        return val

    @cartesian
    def source1(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 0*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)+1
        return val

    @cartesian
    def gradient(self, p):
        """  
        @brief 真解梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -pi*bm.sin(pi*x)*bm.cos(pi*y),
            -pi*bm.cos(pi*x)*bm.sin(pi*y)), axis=-1)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        """
        @brief 真解通量
        """
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)
    
    @cartesian
    def grad_dirichlet(self, p, n):
        val = self.gradient(p)
        if n.ndim == 2:
            return bm.einsum('eqd, ed->eq', val, n)
        else:
            return bm.einsum("eqd,eqd->eq", val, n)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (bm.abs(y - 1.0) < 1e-12) | (bm.abs( y -  0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NF, NQ, 2)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        val = bm.einsum('fqd, fqd -> fq', grad, n) # (NF, NQ)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return bm.abs(x - 1.0) < 1e-12

    @cartesian
    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NE, NQ, 2)
        val = bm.sum(grad*n[:,None,:], axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return bm.abs(x - 0.0) < 1e-12


class LShapeRSinData:
    def __init__(self):
        pass

    def domain(self, hmin=0.1, hmax=0.1, fh=None):
        from fealpy.geometry.domain_2d import LShapeDomain 
        return  LShapeDomain(hmin=hmin, hmax=hmax, fh=fh)

    def init_mesh(self, n=4, meshtype='tri'):
        from fealpy.mesh import TriangleMesh 
        from fealpy.mesh import QuadrangleMesh 
        node = bm.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=bm.float64)
        if meshtype == 'tri':
            cell = bm.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=bm.int32)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            cell = bm.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=bm.int32)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        theta = bm.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        u = (x*x + y*y)**(1/3)*bm.sin(2/3*theta)
        return u

    @cartesian
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        val = bm.zeros(p.shape[:-1], dtype=bm.float64)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        sin = bm.sin
        cos = bm.cos
        pi = bm.pi
        x = p[..., 0]
        y = p[..., 1]
        theta = bm.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        val = bm.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        val[..., 1] = 2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3))
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


