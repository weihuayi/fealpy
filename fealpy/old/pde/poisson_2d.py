import numpy as np

from ..decorator import cartesian, barycentric

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
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """  
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val#-self.solution(p)

    @cartesian
    def gradient(self, p):
        """  
        @brief 真解梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
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
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (np.abs(y - 1.0) < 1e-12) | (np.abs( y -  0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return np.abs(x - 1.0) < 1e-12

    @cartesian
    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return np.abs(x - 0.0) < 1e-12
        

class SinSinData:
    """
        -\\Delta u = f
        u = sin(pi*x)*sin(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        y = p[..., 1]
        return ( y == 1.0) | ( y == 0.0)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return x == 1.0

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa

    @cartesian
    def is_robin_boundary(self, p):
        x = p[..., 0]
        return x == 0.0


class X2Y2Data:
    """
    -\\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        val = x**2*y**2 
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        val = -2*(x**2 + y**2) 
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = 2*x*y**2 
        val[..., 1] = 2*x**2*y 
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa


class TwoHolesData:
    def __init__(self):
        pass

    def domain(self, domaintype='meshpy'):
        if domaintype == 'meshpy':
            from meshpy.triangle import MeshInfo
            domain = MeshInfo()
            points = np.zeros((16, 2), dtype=np.float64)
            points[0:4, :] = np.array(
                    [(0.0, 0.0),
                     (1.0, 0.0),
                     (1.0, 1.0),
                     (0.0, 1.0)], dtype=np.float64)
            idx = np.arange(5, -1, -1)
            points[4:10, 0] = 0.25 + 0.1*np.cos(idx*np.pi/3)
            points[4:10, 1] = 0.75 + 0.1*np.sin(idx*np.pi/3)
            points[10:, 0] = 0.6 + 0.1*np.cos(idx*np.pi/3)
            points[10:, 1] = 0.4 + 0.1*np.sin(idx*np.pi/3)

            facets = np.zeros((16, 2), dtype=np.int_)
            facets[0:4, :] = np.array([(0, 1), (1, 2), (2, 3), (3, 0)],
                    dtype=np.int_)
            facets[4:10, :] = np.array(
                    [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)],
                    dtype=np.int_)
            facets[10:, :] = np.array(
                    [(10, 11), (11, 12), (12, 13), (13, 14),
                        (14, 15), (15, 10)], dtype=np.int_)
            domain.set_points(points)
            domain.set_facets(facets)
            domain.set_holes([(0.25, 0.75), (0.6, 0.4)])
        return domain

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 1 + 10*x**2 + y**2
        return val

    @cartesian
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 0.0 
        return val

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        val = 1.0 
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = 0.0
        val[..., 1] = 0.0
        return val

    @cartesian
    def dirichlet(self, p):
        return 0.0

    @cartesian
    def neumann(self, p):
        """ Neuman  boundary condition
        """
        pass

    @cartesian
    def robin(self, p):
        pass


class ffData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    @cartesian
    def solution(self, p):
        return np.zeros(p.shape[0:-1])

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        val = np.zeros(p.shape, dtype=p.dtype)
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.ones(x.shape, dtype=np.float64)
        I = np.floor(4*x) + np.floor(4*y)
        isMinus = (I % 2 == 0)
        val[isMinus] = - 1
        return val

    @cartesian
    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        val = np.zeros(p.shape[0:-1])
        return val


class KelloggData:
    def __init__(self):
        self.a = 161.4476387975881
        self.b = 1

    def init_mesh(self, n=4, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float64)
        if meshtype == 'tri':
            cell = np.array([
                (1, 4, 0),
                (3, 0, 4),
                (4, 1, 5),
                (2, 5, 1),
                (4, 7, 3),
                (6, 3, 7),
                (7, 4, 8),
                (5, 8, 4)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
        elif meshtype == 'quadtree':
            cell = np.array([
                (0, 1, 4, 3),
                (1, 2, 5, 4),
                (3, 4, 7, 6),
                (4, 5, 8, 7)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
        else:
            raise ValueError("".format)
        return mesh

    @cartesian
    def diffusion_coefficient(self, p):
        idx = (p[..., 0]*p[..., 1] > 0)
        k = np.ones(p.shape[:-1], dtype=np.float64)
        k[idx] = self.a
        return k

    @cartesian
    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        is_subdomain = [p[..., 0]*p[..., 1] > 0, p[..., 0]*p[..., 1] < 0]
        return is_subdomain

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        gamma = 0.1
        sigma = -14.9225565104455152
        rho = pi/4
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)

        mu = ((theta >= 0) & (theta < pi/2))*cos((pi/2-sigma)*gamma)*cos((theta-pi/2+rho)*gamma) \
            + ((theta >= pi/2) & (theta < pi))*cos(rho*gamma)*cos((theta-pi+sigma)*gamma) \
            + ((theta >= pi) & (theta < 1.5*pi))*cos(sigma*gamma)*cos((theta-pi-rho)*gamma) \
            + ((theta >= 1.5*pi) & (theta < 2*pi))*cos((pi/2-rho)*gamma)*cos((theta-1.5*pi-sigma)*gamma)

        u = r**gamma*mu
        return u

    @cartesian
    def gradient(self, p):
        """The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        cos = np.cos
        sin = np.sin
        gamma = 0.1
        sigma = -14.9225565104455152
        rho = pi/4
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        t = 1 + (y/x)**2
        r = np.sqrt(x**2 + y**2)
        rg = r**gamma

        ux1 = ((x >= 0.0) & (y >= 0.0))*(
                gamma*rg*cos((pi/2-sigma)*gamma)*(x*cos((theta-pi/2+rho)*gamma)/(r*r)
                + y*sin((theta-pi/2+rho)*gamma)/(x*x*t))
            )

        uy1 = ((x >= 0.0) & (y >= 0.0))*(gamma*rg*cos((pi/2-sigma)*gamma)*(y*cos((theta-pi/2+rho)*gamma)/(r*r) - sin((theta-pi/2+rho)*gamma)/(x*t)))

        ux2 = ((x <= 0.0) & (y >= 0.0))*(gamma*rg*cos(rho*gamma)*(x*cos((theta-pi+sigma)*gamma)/(r*r) + y*sin((theta-pi+sigma)*gamma)/(x*x*t)))

        uy2 = ((x <= 0.0) & (y >= 0.0))*(gamma*rg*cos(rho*gamma)*(y*cos((theta-pi+sigma)*gamma)/(r*r) - sin((theta-pi+sigma)*gamma)/(x*t)))

        ux3 = ((x <= 0.0) & (y <= 0.0))*(gamma*rg*cos(sigma*gamma)*(x*cos((theta-pi-rho)*gamma)/(r*r)+y*sin((theta-pi-rho)*gamma)/(x*x*t)))

        uy3 = ((x <= 0.0) & (y <= 0.0))*(gamma*rg*cos(sigma*gamma)*(y*cos((theta-pi-rho)*gamma)/(r*r) - sin((theta-pi-rho)*gamma)/(x*t)))

        ux4 = ((x >= 0.0) & (y <= 0.0))*(gamma*rg*cos((pi/2-rho)*gamma)*(x*cos((theta-3*pi/2-sigma)*gamma)/(r*r)+y*sin((theta-3*pi/2-sigma)*gamma)/(x*x*t)))

        uy4 = ((x >= 0.0) & (y <= 0.0))*(gamma*rg*cos((pi/2-rho)*gamma)*(y*cos((theta-3*pi/2-sigma)*gamma)/(r*r)-sin((theta-3*pi/2-sigma)*gamma)/(x*t)))

        val[..., 0] =  ux1+ux2+ux3+ux4
        val[..., 1] =  uy1+uy2+uy3+uy4
        return val

    @cartesian
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p:array object, N*2
        """
        rhs = np.zeros(p.shape[0:-1])
        return rhs

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

class LShapeRSinData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='tri'):
        from fealpy.mesh import TriangleMesh 
        from fealpy.mesh import QuadrangleMesh 
        from fealpy.mesh import Quadtree 
        from fealpy.mesh import Tritree 
        node = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float64)
        if meshtype == 'tri':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int_)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            mesh = Tritree(node, cell)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def domain(self):
        points = [[0, 0], [1, 0], [1, 1], [-1, 1], [-1, -1], [0, -1]]
        facets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        return points, facets

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        u = (x*x + y*y)**(1/3)*np.sin(2/3*theta)
        return u

    @cartesian
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        sin = np.sin
        cos = np.cos
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        val[..., 1] = 2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3))
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


class CrackData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='tri'):
        if meshtype == 'tri':
            node = np.array([
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (1, 0),
                (0, 1)], dtype=np.float64)

            cell = np.array([
                (2, 1, 0),
                (2, 0, 3),
                (2, 5, 1),
                (2, 4, 5)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            r = 1-2**(1/2)/2
            a = 1/2 - 2**(1/2)/2
            rr = 1/2
            node = np.array([
                (0, -1),
                (-rr, -rr),
                (rr, -rr),
                (-r, -r),
                (0, -r),
                (r, -r),
                (-1, 0),
                (-r, 0),
                (0, 0),
                (r, 0),
                (1, 0),
                (r, 0),
                (-r, r),
                (0, r),
                (r, r),
                (-rr, rr),
                (rr, rr),
                (0, 1)], dtype=np.float64)
            cell = np.array([
                (0, 4, 3, 1),
                (2, 5, 4, 0),
                (1, 3, 7, 6),
                (3, 4, 8, 7),
                (4, 5, 9, 8),
                (5, 2, 10, 9),
                (6, 7, 12, 15),
                (7, 8, 13, 12),
                (8, 11, 14, 13),
                (11, 10, 16, 14),
                (12, 13, 17, 15),
                (13, 14, 16, 17)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            node = np.array([
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (1, 0),
                (0, 1)], dtype=np.float64)

            cell = np.array([
                (2, 1, 0),
                (2, 0, 3),
                (2, 5, 1),
                (2, 4, 5)], dtype=np.int_)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def domain(self, n):
        pass 

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]

        r = np.sqrt(x**2 + y**2)
        u = np.sqrt(1/2*(r - x)) - 1/4*r**2
        return u

    @cartesian
    def source(self, p):
        rhs = np.ones(p.shape[0:-1])
        return rhs

    @cartesian
    def gradient(self, p):
        """the gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        r = np.sqrt(x**2 + y**2)
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -0.5*x + (-0.5*x + 0.5*r)**(-0.5)*(0.25*x/r - 0.25)
        val[..., 1] = 0.25*y*(-0.5*x + 0.5*r)**(-0.5)/r - 0.5*y

        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)


class TwoSigularData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float64)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'stri':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    @cartesian
    def solution(self, p):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        t0 = (x + 0.5)**2 + (y - 0.5)**2 + 0.01
        t1 = (x - 0.5)**2 + (y + 0.5)**2 + 0.01
        return 1/t0 - 1/t1

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[..., 0]
        y = p[..., 1]
        t0 = (x + 0.5)**2 + (y - 0.5)**2 + 0.01
        t1 = (x - 0.5)**2 + (y + 0.5)**2 + 0.01
        val = (
                (2*x - 1.0)*(4*x - 2.0)/t1**3 -
                (2*x + 1.0)*(4*x + 2.0)/t0**3 -
                (2*y - 1.0)*(4*y - 2.0)/t0**3 +
                (2*y + 1.0)*(4*y + 2.0)/t1**3 + 4/t0**2 - 4/t1**2
            )

        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        t0 = (x + 0.5)**2 + (y - 0.5)**2 + 0.01
        t1 = (x - 0.5)**2 + (y + 0.5)**2 + 0.01
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -(1.0 - 2*x)/t1**2 + (-2*x - 1.0)/t0**2
        val[..., 1] =(1.0 - 2*y)/t0**2 - (-2*y - 1.0)/t1**2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neuman(self, p):
        """ Neuman  boundary condition
        """
        pass

    @cartesian
    def robin(self, p):
        pass

class CornerSigularData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'stri':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    @cartesian
    def solution(self, p):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        return (x**2 + y**2)**0.2 

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[..., 0]
        y = p[..., 1]
        val = -0.16*(x**2 + y**2)**(-0.8)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = 0.4*x*(x**2 + y**2)**(-0.8)
        val[..., 1] = 0.4*y*(x**2 + y**2)**(-0.8)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neuman(self, p):
        """ Neuman  boundary condition
        """
        pass

    @cartesian
    def robin(self, p):
        pass


class SinSinData:
    def __init__(self):
        pass

    def init_mesh(
            self, n=4, meshtype='quadtree',
            h=0.1,
            nx=10,
            ny=10):
        point = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float64)
        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'stri':
            mesh = StructureQuadMesh([0, 1, 0, 1], nx, ny)
            return mesh
        else:
            raise ValueError("".format)

    @cartesian
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)
        return u

    @cartesian
    def flux(self, p):
        """
        @brief 真解通量
        """
        return -self.gradient(p)

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
        return rhs

    @cartesian
    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float64)
        uprime[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        uprime[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return uprime

class PolynomialData:
    def __init__(self):
        pass

    @cartesian
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        u = (x-x**2)*(y-y**2)
        return u

    @cartesian
    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype == 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float64)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int_)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'halfedge':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'squad':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = 2*(y-y**2)+2*(x-x**2)
        return rhs


    @cartesian
    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float64)
        uprime[..., 0] = (1-2*x)*(y-y**2)
        uprime[..., 1] = (1-2*y)*(x-x**2)
        return uprime

    @cartesian
    def is_boundary(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)


class ExpData:
    def __init__(self):
        pass

    @cartesian
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        u = np.exp(x**2+y**2)
        return u

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = -(4*x**2+4*y**2+4)*(np.exp(x**2+y**2))
        return rhs


    @cartesian
    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float64)
        uprime[..., 0] = 2*x*(np.exp(x**2+y**2))
        uprime[..., 1] = 2*y*(np.exp(x**2+y**2))
        return uprime

    @cartesian
    def is_boundary(self, p):
        eps = 1e-14 
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)

class ArctanData:
    def __init__(self):
        pass

    @cartesian
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        u = np.arctan((x**2+y**2)*100)
        return u

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        a = 10000*(x**2+y**2)**2
        rhs = -100*(4-4*a)/(a+1)**2
        return rhs

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        a = 10000*(x**2+y**2)**2
        uprime = np.zeros(p.shape, dtype=np.float64)
        uprime[..., 0] = 200*x/(1+a)
        uprime[..., 1] = 200*y/(1+a)
        return uprime

    def is_boundary(self, p):
        eps = 1e-14 
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)



class CircleSinSinData():
    def __init__(self):
        from fealpy.geometry import CircleCurve
        self.curve = CircleCurve()

    def domain(self):
        return self.curve

    
    def init_mesh(self, n=0):
        t = self.curve.radius
        c = np.sqrt(3.0)/2.0
        node = np.array([
            [0.0, 0.0],
            [  t, 0.0],
            [t/2.0, c*t],
            [-t/2.0, c*t],
            [-t, 0.0],
            [-t/2.0, -c*t],
            [t/2.0, -c*t]],dtype=np.float64)
        cell = np.array([
            [0,1,2],
            [0,2,3],
            [0,3,4],
            [0,4,5],
            [0,5,6],
            [0,6,1]],dtype=np.int_)
    
        mesh = TriangleMesh(node, cell)
        for i in range(n):
            NN = mesh.number_of_nodes()
            mesh.uniform_refine()
            node = mesh.entity('node')
            isBdNode = mesh.ds.boundary_node_flag()
            node[isBdNode], _= self.curve.project(node[isBdNode])
        return mesh

    @cartesian
    def solution(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)
        return u
    
    @cartesian
    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        f = 2*pi**2*np.sin(pi*x)*np.sin(pi*y)
        return f
    
    @cartesian
    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        grad = np.zeros(p.shape,dtype=np.float_)
        grad[...,0] = pi*np.cos(pi*x)*np.sin(pi*y)
        grad[...,1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return grad

    @cartesian
    def dirichlet(self,p):
        p, _ = self.curve.project(p)
        return self.solution(p)

