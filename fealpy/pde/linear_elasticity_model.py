import numpy as np

from ..decorator  import cartesian 
from ..mesh import TriangleMesh


class LinearElasticityTempalte():
    def __init__(self):
        pass

    def domain(self):
        pass

    def init_mesh(self):
        pass

    @cartesian
    def displacement(self, p):
        pass

    @cartesian
    def jacobian(self, p):
        pass

    @cartesian
    def strain(self, p):
        pass

    @cartesian
    def stress(self, p):
        pass

    @cartesian
    def source(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p):
        pass

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neuman_boundary(self, p):
        pass

    @cartesian
    def is_fracture_boundary(self, p):
        pass


class BoxDomainData2d():
    def __init__(self, E=1e+5, nu=0.2):
        self.E = E 
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

    def domain(self):
        return [0, 1, 0, 1]

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh 

    @cartesian
    def displacement(self, p):
        return 0.0

    @cartesian
    def jacobian(self, p):
        return 0.0

    @cartesian
    def strain(self, p):
        return 0.0

    @cartesian
    def stress(self, p):
        return 0.0

    @cartesian
    def source(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def neumann(self, p, n):
        val = np.array([-500, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x) < 1e-13
        return flag

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x - 1) < 1e-13
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        pass

class BoxDomainData3d():
    def __init__(self):
        self.L = 1
        self.W = 0.2

        self.mu = 1
        self.rho = 1

        delta = self.W/self.L
        gamma = 0.4*delta**2
        beta = 1.25

        self.lam = beta
        self.g = gamma
        self.d = np.array([0.0, 0.0, -1.0])

    def domain(self):
        return [0.0, self.L, 0.0, self.W, 0.0, self.W]

    def init_mesh(self, n=1):
        from fealpy.mesh import MeshFactory as MF 
        i = 2**n
        domain = self.domain()
        mesh = MF.boxmesh3d(domain, nx=5*i, ny=1*i, nz=1*i, meshtype='tet')
        return mesh

    @cartesian
    def displacement(self, p):
        pass

    @cartesian
    def jacobian(self, p):
        pass

    @cartesian
    def strain(self, p):
        pass

    @cartesian
    def stress(self, p):
        pass

    @cartesian
    def source(self, p):
        shape = len(p.shape[:-1])*(1,) + (-1, )
        val = self.d*self.g*self.rho
        return val.reshape(shape) 
    @cartesian
    def dirichlet(self, p):
        shape = len(p.shape)*(1, )
        val = np.array([0.0])
        return val.reshape(shape)

    @cartesian
    def is_dirichlet_boundary(self, p):
        return np.abs(p[..., 0]) < 1e-12

class BeamData2d():
    def __init__(self, E = 2*10**6, nu = 0.3):
        self.l = 48
        self.h = 12
        self.q = 100

        self.nu = nu
        self.E = E
       
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))       

    def domain(self):
        return [0.0, self.l, -self.h/2.0, self.h/2.0]

    def init_mesh(self, n=1):
        from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
        domain = self.domain()
        mesh = rectangledomainmesh(domain, nx=4*n, ny=1*n, meshtype='tri')
        return mesh

    @cartesian
    def displacement(self, p):
        q = self.q
        l = self.l
        h = self.h
        
        nu = self.nu
        E = self.E
        I = h**3/12
         
        x = p[..., 0]
        y = p[..., 1]
        
        val = np.zeros_like(p)

        val[..., 0] = -q*x*y*(l-x)*(l-2*x)/(12*E*I)
        val[..., 0] += q*(l-2*x)*y*(3*(1+nu)*h**2 - 2*(2+nu)*y**2)/(24*E*I)
        
        val[..., 1] = q*(l-x)**2*x**2/(24*E*I)
        val[..., 1] += q*(-2*(1+2*nu)*y**4 + ((6*x**2-6*l*x+l**2)*2*nu+3*h**2*(nu+1)**2)*y**2 + 24*I*(nu**2-1)*y)/(48*E*I)
                
        return val
        
    @cartesian
    def stress(self, p):
        q = self.q
        l = self.l
        h = self.h
        
        nu = self.nu
        E = self.E
        I = h**3/12    
        x = p[..., 0]
        y = p[..., 1]
        
        shape = p.shape[:-1] + (2, 2)
        val = np.zeros(shape, dtype=np.float)
        
        val[..., 0, 0] += -q*(x**2-l*x+l**2/6)*y/(2*I)    
        val[..., 0, 0] += q*(8*y**3-3*(2+nu)*h**2*y-nu*h**3)/(24*I)  
        
        val[..., 0, 1] += q*(l-2*x)(h**2/4-y**2)/(4*I) 
        
        val[..., 1, 0] += q*(l-2*x)(h**2/4-y**2)/(4*I) 
        
        val[..., 1, 1] += -q*(4*y**3-3*h**2*y+h**3)/(24*I)
        return val
        
    @cartesian
    def source(self, p):
        val = np.zeros_like(p)
        return val 
        
    @cartesian
    def dirichlet(self, p):  
        val = np.zeros_like(p) 
        return val
        
    @cartesian    
    def neumann(self, p, n):  # p 是受到面力的节点坐标
        val = np.array([0.0, -self.q], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)
        
    @cartesian
    def is_dirichlet_boundary(self, p):
        eps = 1e-10
        return (np.abs(p[..., 0]) < eps) | (np.abs(p[..., 0]-self.l) < eps)
        
    @cartesian        
    def is_neumann_boundary(self, p):
        eps = 1e-10
        return (np.abs(p[..., 1]-self.h/2) < eps)

class LShapeDomainData2d():
    def __init__(self, E=1e+5, nu=0.499):

        self.E = E 
        self.nu = nu

        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

        self.alpha = 0.544483736782

        alpha = self.alpha
        omega = 3*np.pi/4
        lam = self.lam
        mu = self.mu
        cos = np.cos
        sin = np.sin

        self.C1 = -cos((alpha + 1)*omega)/cos((alpha - 1)*omega)
        self.C2 = 2*(lam + 2*mu)/(lam + mu)

    def domain(self, domaintype='meshpy'):
        if domaintype == 'meshpy':
            from meshpy.triangle import MeshInfo
            domain = MeshInfo()
            points = np.array([
                (0, 0), (-1, -1), (0, -2), ( 1, -1),
                (2, 0), ( 1,  1), (0,  2), (-1,  1)], dtype=np.float)
            facets = np.array([
                (0, 1), (1, 2), (2, 3), (3, 4), 
                (4, 5), (5, 6), (6, 7), (7, 0)], dtype=np.int)
            domain.set_points(points)
            domain.set_facets(facets)
            return domain
        if domaintype == 'halfedge':
            return None

    def init_mesh(self, n=2, meshtype='tri', h=None):
        """ generate the initial mesh
        """
        from fealpy.mesh import TriangleMesh
        node = np.array([
            (0, 0), (-1, -1), (0, -2), ( 1, -1),
            (2, 0), ( 1,  1), (0,  2), (-1,  1)], dtype=np.float)
        cell = np.array([
            (1, 2, 0), (3, 0, 2), (4, 5, 3),
            (0, 3, 5), (5, 6, 0), (7, 0, 6)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n=n)
        return mesh

    def displacement(self, p):
        alpha = self.alpha
        lam = self.lam
        mu = self.mu
        C1 = self.C1
        C2 = self.C2

        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin

        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = np.sqrt(x**2 + y**2)
        val = np.zeros_like(p)

        val[..., 0]  = -(alpha + 1)*cos((alpha+1)*theta)
        val[..., 0] += (C2 - alpha - 1)*C1*cos((alpha -1)*theta)
        val[..., 0] /= 2*mu
        val[..., 0] *= r**alpha

        val[..., 1]  = (alpha + 1)*sin((alpha+1)*theta)
        val[..., 1] += (C2 + alpha - 1)*C1*sin((alpha -1)*theta)
        val[..., 1] /= 2*mu
        val[..., 1] *= r**alpha
        return val

    def jacobian(self, p):
        alpha = self.alpha
        lam = self.lam
        mu = self.mu
        C1 = self.C1
        C2 = self.C2

        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin

        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = np.sqrt(x**2 + y**2)

        shape = p.shape[:-1] + (2, 2)
        val = np.zeros(shape, dtype=p.dtype)

        c0 = cos(theta*(alpha - 1))
        c1 = cos(theta*(alpha + 1))
        s0 = sin(theta*(alpha - 1))
        s1 = sin(theta*(alpha + 1))
        t = r**alpha/(2*mu*r**2)

        val[..., 0, 0] += alpha*x*(C1*(C2 - alpha - 1)*c0 + (-alpha - 1)*c1)*t 
        val[..., 0, 0] += (C1*y*(alpha - 1)*(C2 - alpha - 1)*s0
                + y*(-alpha - 1)*(alpha + 1)*s1)*t

        val[..., 0, 1] += alpha*y*(C1*(C2 - alpha - 1)*c0 + (-alpha - 1)*c1)*t
        val[..., 0, 1] += (-C1*x*(alpha - 1)*(C2 - alpha - 1)*s0 
                - x*(-alpha - 1)*(alpha + 1)*s1)*t

        val[..., 1, 0] += alpha*x*(C1*(C2 + alpha - 1)*s0 + (alpha + 1)*s1)*t
        val[..., 1, 0] += (-C1*y*(alpha - 1)*(C2 + alpha - 1)*c0
            - y*(alpha + 1)**2*c1)*t

        val[..., 1, 1] += alpha*y*(C1*(C2 + alpha - 1)*s0 + (alpha + 1)*s1)*t 
        val[..., 1, 1] += (C1*x*(alpha - 1)*(C2 + alpha - 1)*c0
            + x*(alpha + 1)**2*c1)*t
        return val
        
    def strain(self, p):
        val = self.jacobian(p)
        t = (val[..., 0, 1] + val[..., 1, 0])/2
        val[..., 0, 1] = t
        val[..., 1, 0] = t
        return val

    def stress(self, p):
        lam = self.lam
        mu = sefl.mu

        val = self.strain(p)
        t = val.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        val *= 2*mu
        idx = np.arange(2)
        val[..., idx, idx] += t
        return val

    def source(self, p):
        val = np.zeros_like(p)
        return val

    def dirichlet(self, p):
        val = self.displacement(p)
        return val

    def neumann(self, p, n):
        val = np.zeros_like(p)
        return val

    def is_dirichlet_boundary(self, p):
        val = self.is_neuman_boundary(p)
        return ~val

    def is_neuman_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.abs(np.abs(x) - np.abs(y)) < 1e-12
        return val

       

class CookMembraneData():
    def __init__(self, E=1e+5, nu=0.3):
        self.E = E
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

    def domain(self, domaintype='meshpy'):
        if domaintype == 'meshpy':
            from meshpy.triangle import MeshInfo
            domain = MeshInfo()
            points = np.array([
                (0, 0), (48, 44), (48, 60), (0, 44)], dtype=np.float)
            facets = np.array([
                (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)
            domain.set_points(points)
            domain.set_facets(facets)
            return domain
        if domaintype == 'halfedge':
            return None

    def init_mesh(self, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        from meshpy.triangle import build
        domain = self.domain()
        mesh = build(domain, max_volume=h**2)
        node = np.array(mesh.points, dtype=np.float)
        cell = np.array(mesh.elements, dtype=np.int)
        if meshtype == 'tri':
            mesh = TriangleMesh(node, cell)
            return mesh 

    def displacement(self, p):
        return None

    def strain(self, p):
        return None

    def stress(self, p):
        return None

    def source(self, p):
        val = np.zeros(p.shape, dtype=p.dtype)
        return val 

    def neumann(self, p, n):  # p 是受到面力的节点坐标
        """
        Neuman  boundary condition
        p: (NQ, NE, 2)
        n: (NE, 2)
        """
        val = np.zeros_like(p)
        val[..., 1] = 1
        return val

    def dirichlet(self, p):  
        """
        """
        val = np.zeros_like(p) 
        return val

    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        return  np.abs(x) < 1e-12

    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return np.abs(x - 48) < 1e-12

class CantileverBeam2d():
    def __init__(self, E=3e+7, nu=0.3, P=1000, L=48, W=12):
        self.E = E
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

        self.L = L  # length
        self.W = W  # width 
        self.P = P  # load at the right end
        self.I = self.W**3/12

    def domain(self):
        return [0, self.L, -self.W/2, self.W/2]

    def init_mesh(self, n=1):
        from fealpy.mesh.simple_mesh_generator import  rectangledomainmesh
        box = self.domain()
        mesh = rectangledomainmesh(box, nx=8, ny=2)
        mesh.uniform_refine(n)
        return mesh

    def displacement(self, p):
        L = self.L
        P = self.P
        W = self.W
        I = self.I
        E = self.E
        nu = self.nu

        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = P*y/(6*E*I)*((6*L-3*x)*x+(2+nu)*(y**2 - W**2/4))
        val[..., 1] = -P/(6*E*I)*(3*nu*y**2*(L-x)+(4+5*nu)*W**2*x/4+(3*L-x)*x**2)
        return val

    def strain(self, p):
        pass

    def stress(self, p):
        P = self.P
        L = self.L
        I = self.I
        W = self.W

        x = p[..., 0]
        y = p[..., 1]
        shape = p.shape[:-1] + (2, 2)
        val = np.zeros(shape, dtype=np.float)
        val[..., 0, 0] = P*(L-x)*y/I
        val[..., 0, 1] = -P/(2*I)*(W**2/4 - y**2)
        val[..., 1, 0] = val[..., 0, 1] 
        return val

    def source(self, p):
        val = np.zeros(p.shape, dtype=p.dtype)
        return val

    def neumann(self, p, n):  # p 是受到面力的节点坐标
        """
        Neuman  boundary condition
        p: (NQ, NE, 2)
        n: (NE, 2)
        """
        val = self.stress(p)
        val = np.einsum('...ijk, ik->...ij', val, n)
        return val

    def dirichlet(self, p):  
        """
        """
        val = self.displacement(p) 
        return val

    def is_dirichlet_boundary(self, p):
        return  np.abs(p[..., 0]) < 1e-12

    def is_neumann_boundary(self, p):
        return np.abs(p[..., 0] - self.L) < 1e-12


class QiModel3d():
    def __init__(self, lam=1.0, mu=0.5):
        self.lam = lam
        self.mu = mu
    def init_mesh(self, n=2):
        from ..mesh import TetrahedronMesh
        node = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]], dtype=np.float)

        cell = np.array([
            [0,1,2,6],
            [0,5,1,6],
            [0,4,5,6],
            [0,7,4,6],
            [0,3,7,6],
            [0,2,3,6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def displacement(self, p):
        mu = self.mu
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 200*mu*(x - x**2)**2*(2*y**3 - 3*y**2 + y)*(2*z**3 - 3*z**2 + z)  
        val[..., 1] = -100*mu*(y - y**2)**2*(2*x**3 - 3*x**2 + x)*(2*z**3 - 3*z**2 + z)  
        val[..., 2] = -100*mu*(z - z**2)**2*(2*y**3 - 3*y**2 + y)*(2*x**3 - 3*x**2 + x)  
        val = np.einsum('...jk, k->...jk', val, np.array([2**4, 2**5, 2**6]))
        print('val.shape:', val.shape)
        return val

    def grad_displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        a, b, c = 2**4, 2**5, 2**6

        shape = p.shape + (3, )
        val = np.zeros(shape, dtype=np.float)
        t0 = (-x + 1)*(-y + 1)*(-z + 1)
        t1 = x*y*z
        val[..., 0, 0] = -a*t1*(-y + 1)*(-z + 1) + a*y*z*t0
        val[..., 0, 1] = -a*t1*(-x + 1)*(-z + 1) + a*x*z*t0
        val[..., 0, 2] = -a*t1*(-x + 1)*(-y + 1) + a*x*y*t0
        val[..., 1, 0] = -b*t1*(-y + 1)*(-z + 1) + b*y*z*t0
        val[..., 1, 1] = -b*t1*(-x + 1)*(-z + 1) + b*x*z*t0
        val[..., 1, 2] = -b*t1*(-x + 1)*(-y + 1) + b*x*y*t0
        val[..., 2, 0] = -c*t1*(-y + 1)*(-z + 1) + c*y*z*t0
        val[..., 2, 1] = -c*t1*(-x + 1)*(-z + 1) + c*x*z*t0
        val[..., 2, 2] = -c*t1*(-x + 1)*(-y + 1) + c*x*y*t0
        return val

    def stress(self, p):
        lam = self.lam
        mu = self.mu
        du = self.grad_displacement(p)
        val = mu*(du + du.swapaxes(-1, -2))
        idx = np.arange(3)
        val[..., idx, idx] += lam*du.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        return val
        

    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:3], axis=-1)
        aphi[..., 0:3] -= lam/(2*mu+3*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi

    def div_stress(self, p):
        return -self.source(p)

    def source(self, p):
        lam = self.lam
        mu = self.mu
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        a, b, c = 2**4, 2**5, 2**6
        val = np.zeros(p.shape, dtype=np.float)
        t0 = (-x+1)*(-y+1)*(-z+1) 
        t1 = (-y+1)*(-z+1)
        t2 = (-x+1)*(-z+1)
        t3 = (-x+1)*(-y+1)
        val[..., 0] += 4*a*mu*t1*y*z 
        val[..., 0] -= lam*(-2*a*t1*y*z + b*t0*z - b*t1*x*z - b*t2*y*z + b*x*y*z*(-z + 1) + c*t0*y - c*t1*x*y - c*t3*y*z + c*x*y*z*(-y + 1)) 
        val[..., 0] -= mu*(-2*a*t2*x*z + b*t0*z - b*t1*x*z - b*t2*y*z + b*x*y*z*(-z + 1)) 
        val[..., 0] -= mu*(-2*a*t3*x*y + c*t0*y - c*t1*x*y - c*t3*y*z + c*x*y*z*(-y + 1))

        val[..., 1] += 4*b*mu*t2*x*z 
        val[..., 1] -= lam*(a*t0*z - a*t1*x*z - a*t2*y*z + a*x*y*z*(-z + 1) - 2*b*t2*x*z + c*t0*x - c*t2*x*y - c*t3*x*z + c*x*y*z*(-x + 1)) 
        val[..., 1] -= mu*(a*t0*z - a*t1*x*z - a*t2*y*z + a*x*y*z*(-z + 1) - 2*b*t1*y*z) 
        val[..., 1] -= mu*(-2*b*t3*x*y + c*t0*x - c*t2*x*y - c*t3*x*z + c*x*y*z*(-x + 1))

        val[..., 2] += 4*c*mu*t3*x*y 
        val[..., 2] -= lam*(a*t0*y - a*t1*x*y - a*t3*y*z + a*x*y*z*(-y + 1) + b*t0*x - b*t2*x*y - b*t3*x*z + b*x*y*z*(-x + 1) - 2*c*t3*x*y)
        val[..., 2] -= mu*(a*t0*y - a*t1*x*y - a*t3*y*z + a*x*y*z*(-y + 1) - 2*c*t1*y*z) 
        val[..., 2] -= mu*(b*t0*x - b*t2*x*y - b*t3*x*z + b*x*y*z*(-x + 1) - 2*c*t2*x*z)
        return val


class PolyModel3d():
    def __init__(self, lam=1.0, mu=0.5):
        self.lam = lam
        self.mu = mu
    def init_mesh(self, n=2):
        from ..mesh import TetrahedronMesh
        node = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1], 
            [1, 1, 1],
            [0, 1, 1]], dtype=np.float) 

        cell = np.array([
            [0,1,2,6],
            [0,5,1,6],
            [0,4,5,6],
            [0,7,4,6],
            [0,3,7,6],
            [0,2,3,6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    @cartesian
    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = x*(1-x)*y*(1-y)*z*(1-z) 
        val = np.einsum('...j, k->...jk', val, np.array([2**4, 2**5, 2**6]))
        return val

    @cartesian
    def grad_displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        a, b, c = 2**4, 2**5, 2**6

        shape = p.shape + (3, )
        val = np.zeros(shape, dtype=np.float)
        t0 = (-x + 1)*(-y + 1)*(-z + 1)
        t1 = x*y*z
        val[..., 0, 0] = -a*t1*(-y + 1)*(-z + 1) + a*y*z*t0
        val[..., 0, 1] = -a*t1*(-x + 1)*(-z + 1) + a*x*z*t0
        val[..., 0, 2] = -a*t1*(-x + 1)*(-y + 1) + a*x*y*t0
        val[..., 1, 0] = -b*t1*(-y + 1)*(-z + 1) + b*y*z*t0
        val[..., 1, 1] = -b*t1*(-x + 1)*(-z + 1) + b*x*z*t0
        val[..., 1, 2] = -b*t1*(-x + 1)*(-y + 1) + b*x*y*t0
        val[..., 2, 0] = -c*t1*(-y + 1)*(-z + 1) + c*y*z*t0
        val[..., 2, 1] = -c*t1*(-x + 1)*(-z + 1) + c*x*z*t0
        val[..., 2, 2] = -c*t1*(-x + 1)*(-y + 1) + c*x*y*t0
        return val

    @cartesian
    def stress(self, p):
        lam = self.lam
        mu = self.mu
        du = self.grad_displacement(p)
        Au = (du + du.swapaxes(-1, -2))/2
        val = 2*mu*Au
        val[..., np.arange(3), np.arange(3)] += lam*Au.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        return val
        

    @cartesian
    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:3], axis=-1)
        aphi[..., 0:3] -= lam/(2*mu+3*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi

    @cartesian
    def div_stress(self, p):
        return -self.source(p)

    @cartesian
    def source(self, p):
        lam = self.lam
        mu = self.mu
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        a, b, c = 2**4, 2**5, 2**6
        val = np.zeros(p.shape, dtype=np.float)
        t0 = (-x+1)*(-y+1)*(-z+1) 
        t1 = (-y+1)*(-z+1)
        t2 = (-x+1)*(-z+1)
        t3 = (-x+1)*(-y+1)
        val[..., 0] += 4*a*mu*t1*y*z 
        val[..., 0] -= lam*(-2*a*t1*y*z + b*t0*z - b*t1*x*z - b*t2*y*z + b*x*y*z*(-z + 1) + c*t0*y - c*t1*x*y - c*t3*y*z + c*x*y*z*(-y + 1)) 
        val[..., 0] -= mu*(-2*a*t2*x*z + b*t0*z - b*t1*x*z - b*t2*y*z + b*x*y*z*(-z + 1)) 
        val[..., 0] -= mu*(-2*a*t3*x*y + c*t0*y - c*t1*x*y - c*t3*y*z + c*x*y*z*(-y + 1))

        val[..., 1] += 4*b*mu*t2*x*z 
        val[..., 1] -= lam*(a*t0*z - a*t1*x*z - a*t2*y*z + a*x*y*z*(-z + 1) - 2*b*t2*x*z + c*t0*x - c*t2*x*y - c*t3*x*z + c*x*y*z*(-x + 1)) 
        val[..., 1] -= mu*(a*t0*z - a*t1*x*z - a*t2*y*z + a*x*y*z*(-z + 1) - 2*b*t1*y*z) 
        val[..., 1] -= mu*(-2*b*t3*x*y + c*t0*x - c*t2*x*y - c*t3*x*z + c*x*y*z*(-x + 1))

        val[..., 2] += 4*c*mu*t3*x*y 
        val[..., 2] -= lam*(a*t0*y - a*t1*x*y - a*t3*y*z + a*x*y*z*(-y + 1) + b*t0*x - b*t2*x*y - b*t3*x*z + b*x*y*z*(-x + 1) - 2*c*t3*x*y)
        val[..., 2] -= mu*(a*t0*y - a*t1*x*y - a*t3*y*z + a*x*y*z*(-y + 1) - 2*c*t1*y*z) 
        val[..., 2] -= mu*(b*t0*x - b*t2*x*y - b*t3*x*z + b*x*y*z*(-x + 1) - 2*c*t2*x*z)
        return val

    @cartesian
    def dirichlet(self, p):  
        """
        """
        val = self.displacement(p) 
        return val
        
    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:,2] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps) | (p[:, 2] > 1.0 - eps)

class HuangModel2d():
    def __init__(self, lam=10, mu=1):
        self.lam = lam
        self.mu = mu

    def init_mesh(self, n=4):
        from ..mesh import TriangleMesh
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh 

    @cartesian
    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi/2*np.sin(pi*x)**2*np.sin(2*pi*y) 
        val[..., 1] = -pi/2*np.sin(pi*y)**2*np.sin(2*pi*x) 
        return val

    @cartesian
    def grad_displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]

        sin = np.sin
        cos = np.cos
        pi = np.pi

        shape = p.shape + (2, )
        val = np.zeros(shape, dtype=np.float)
        val[..., 0, 0] = pi**2*sin(pi*x)*sin(2*pi*y)*cos(pi*x)
        val[..., 0, 1] = pi**2*sin(pi*x)**2*cos(2*pi*y)
        val[..., 1, 0] = -pi**2*sin(pi*y)**2*cos(2*pi*x)
        val[..., 1, 1] = -pi**2*sin(2*pi*x)*sin(pi*y)*cos(pi*y)
        return val

    @cartesian
    def stress(self, p):
        lam = self.lam
        mu = self.mu
        du = self.grad_displacement(p)
        Au = (du + du.swapaxes(-1, -2))/2
        val = 2*mu*Au
        val[..., range(2), range(2)] += lam*Au.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        return val
        

    @cartesian
    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:2], axis=-1)
        aphi[..., 0:2] -= lam/(2*mu+2*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi

    @cartesian
    def div_stress(self, p):
        return -self.source(p)

    @cartesian
    def source(self, p):
        lam = self.lam
        mu = self.mu
        x = p[..., 0]
        y = p[..., 1]

        sin = np.sin
        cos = np.cos
        pi = np.pi


        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] -= lam*(-pi**3*sin(pi*x)**2*sin(2*pi*y) - 2*pi**3*sin(pi*y)*cos(2*pi*x)*cos(pi*y) + pi**3*sin(2*pi*y)*cos(pi*x)**2)
        val[..., 0] -= 2*mu*(-pi**3*sin(pi*x)**2*sin(2*pi*y) - pi**3*sin(pi*y)*cos(2*pi*x)*cos(pi*y))
        val[..., 0] += 2*pi**3*mu*sin(pi*x)**2*sin(2*pi*y) - 2*pi**3*mu*sin(2*pi*y)*cos(pi*x)**2

        val[..., 1] -= lam*(2*pi**3*sin(pi*x)*cos(pi*x)*cos(2*pi*y) + pi**3*sin(2*pi*x)*sin(pi*y)**2 - pi**3*sin(2*pi*x)*cos(pi*y)**2)
        val[..., 1] -= 2*mu*(pi**3*sin(pi*x)*cos(pi*x)*cos(2*pi*y) + pi**3*sin(2*pi*x)*sin(pi*y)**2)
        val[..., 1] -= 2*pi**3*mu*sin(2*pi*x)*sin(pi*y)**2 
        val[..., 1] += 2*pi**3*mu*sin(2*pi*x)*cos(pi*y)**2

#        val[..., 0] = -pi**3*sin(2*pi*y)*(2*cos(2*pi*x) - 1)
#        val[..., 1] = pi**3*sin(2*pi*x)*(2*cos(2*pi*y) - 1)

        return val

    @cartesian
    def dirichlet(self, p):  
        """
        """
        val = self.displacement(p) 
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)


class Model2d():
    def __init__(self, lam=1.0, mu=0.5):
        self.lam = lam
        self.mu = mu

    def init_mesh(self, n=4):
        from ..mesh import TriangleMesh
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh 

    @cartesian
    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = np.exp(x - y)*x*(1 - x)*y*(1 - y)
        val[..., 1] = np.sin(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def grad_displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]

        sin = np.sin
        cos = np.cos
        pi = np.pi
        e = np.exp(x - y)

        shape = p.shape + (2, )
        val = np.zeros(shape, dtype=np.float)
        val[..., 0, 0] = e*(x*y*(-x + 1)*(-y + 1) - x*y*(-y + 1) + y*(-x + 1)*(-y + 1))
        val[..., 0, 1] = e*(-x*y*(-x + 1)*(-y + 1) - x*y*(-x + 1) + x*(-x + 1)*(-y + 1))
        val[..., 1, 0] = pi*sin(pi*y)*cos(pi*x)
        val[..., 1, 1] = pi*sin(pi*x)*cos(pi*y)
        return val

    @cartesian
    def stress(self, p):
        lam = self.lam
        mu = self.mu
        du = self.grad_displacement(p)
        Au = (du + du.swapaxes(-1, -2))/2
        val = 2*mu*Au
        val[..., range(2), range(2)] += lam*Au.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        return val
        

    @cartesian
    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:2], axis=-1)
        aphi[..., 0:2] -= lam/(2*mu+2*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi

    @cartesian
    def div_stress(self, p):
        return -self.source(p)

    @cartesian
    def source(self, p):
        lam = self.lam
        mu = self.mu
        x = p[..., 0]
        y = p[..., 1]

        sin = np.sin
        cos = np.cos
        pi = np.pi


        ss = sin(pi*x)*sin(pi*y)
        cc = cos(pi*x)*cos(pi*y)
        e = np.exp(x - y)
        t0 = (-x + 1)*(-y + 1)

        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] -= lam*(pi**2*cc + e*t0*x*y + 2*e*t0*y - 2*e*x*y*(-y + 1) - 2*e*y*(-y + 1)) 
        val[..., 0] -= 2*mu*(e*t0*x*y + 2*e*t0*y - 2*e*x*y*(-y + 1) - 2*e*y*(-y + 1)) 
        val[..., 0] -= 2*mu*(pi**2*cc/2 + e*t0*x*y/2 - e*t0*x + e*x*y*(-x + 1) - e*x*(-x + 1))

        val[..., 1] -= lam*(-e*t0*x*y + e*t0*x - e*t0*y + e*t0 - e*x*y*(-x + 1) + e*x*y*(-y + 1) + e*x*y - e*x*(-y + 1) - e*y*(-x + 1) - pi**2*ss)
        val[..., 1] += 2*pi**2*mu*ss 
        val[..., 1] -= 2*mu*(-e*t0*x*y/2 + e*t0*x/2 - e*t0*y/2 + e*t0/2 - e*x*y*(-x + 1)/2 + e*x*y*(-y + 1)/2 + e*x*y/2 - e*x*(-y + 1)/2 - e*y*(-x + 1)/2 - pi**2*ss/2)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.displacement(p)

class Hole2d():
    def __init__(self, lam=1.0, mu=0.5):
        self.lam = lam
        self.mu = mu

    def init_mesh(self, n=2):
        from ..mesh import TriangleMesh
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)

        NN = mesh.number_of_nodes()
        node = np.zeros((NN+3, 2), dtype=np.float64)
        node[:NN] = mesh.entity('node')
        node[NN:] = node[[5], :]
        cell = mesh.entity('cell')

        cell[13][cell[13] == 5] = NN
        cell[18][cell[18] == 5] = NN

        cell[19][cell[19] == 5] = NN+1
        cell[12][cell[12] == 5] = NN+1

        cell[6][cell[6] == 5] = NN+2

        return  TriangleMesh(node, cell)

    @cartesian
    def dirichlet(self, p):
        val = np.zeros(p.shape, dtype=p.dtype)
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        return  np.abs(p[..., 0]) < -2 + 1e-12

    @cartesian
    def neumann(self, p, n):
        val = np.zeros_like(p)
        val[..., 1] = -1
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        return  np.abs(p[..., 0]) > 2 + 1e-12

    @cartesian
    def source(self, p):
        NN = int(len(p)/2)
        val = np.zeros_like(p)
        val[5, 1] = 1
        val[NN, 1] = 1
        val[NN+1, 1] = 1
        val[NN+2, 1] = 1
        return val

    @cartesian
    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float) 
        
        lam = self.lam
        mu = self.mu
        nu = lam/(2*(lam + mu))
        
        P = 100
        L = 4
        W = 4
        I = W**3/12
        val[..., 0] = P*y/(6*E*I)*((6*L-3*x)*x+(2+nu)*(y**2 - W**2/4))
        val[..., 1] = -P/(6*E*I)*(3*nu*y**2*(L-x)+(4+5*nu)*W**2*x/4+(3*L-x)*x**2)
        return val
