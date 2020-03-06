import numpy as np

class CantileverBeam2d():
    def __init__(self, E=3*10**7, nu=0.3, P=1000, L=48, W=12):
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

    def stress(self, p):
        P = self.P
        L = self.L
        I = self.I
        W = self.W

        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros((len(x), 2, 2), dtype=np.float)
        val[..., 0, 0] = P*(L-x)*y/I
        val[..., 0, 1] = -P/(2*I)*(W**2/4 - y**2)
        val[..., 1, 0] = -P/(2*I)*(W**2/4 - y**2)
        return val

    def source(self, p):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    def neuman(self, p, n):  # p 是受到面力的节点坐标
        """
        Neuman  boundary condition
        p: (NQ, NE, 2)
        n: (NE, 2)
        """
        W = self.W
        P = self.P
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 1] = (y + W/2)*(y - W/2)*P
        return val

    def dirichlet(self, p):  
        """
        """
        val = np.zeros(p.shape, dtype=np.float)
        return val

    def is_dirichlet_boundary(self, p):
        return  np.abs(p[..., 0]) < 1e-12

    def is_neuman_boundary(self, p):
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

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = x*(1-x)*y*(1-y)*z*(1-z) 
        val = np.einsum('...j, k->...jk', val, np.array([2**4, 2**5, 2**6]))
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
        Au = (du + du.swapaxes(-1, -2))/2
        val = 2*mu*Au
        val[..., np.arange(3), np.arange(3)] += lam*Au.trace(axis1=-2, axis2=-1)[..., np.newaxis]
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

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi/2*np.sin(pi*x)**2*np.sin(2*pi*y) 
        val[..., 1] = -pi/2*np.sin(pi*y)**2*np.sin(2*pi*x) 
        return val

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

    def stress(self, p):
        lam = self.lam
        mu = self.mu
        du = self.grad_displacement(p)
        Au = (du + du.swapaxes(-1, -2))/2
        val = 2*mu*Au
        val[..., range(2), range(2)] += lam*Au.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        return val
        

    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:2], axis=-1)
        aphi[..., 0:2] -= lam/(2*mu+2*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi

    def div_stress(self, p):
        return -self.source(p)

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

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = np.exp(x - y)*x*(1 - x)*y*(1 - y)
        val[..., 1] = np.sin(pi*x)*np.sin(pi*y)
        return val

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

    def stress(self, p):
        lam = self.lam
        mu = self.mu
        du = self.grad_displacement(p)
        Au = (du + du.swapaxes(-1, -2))/2
        val = 2*mu*Au
        val[..., range(2), range(2)] += lam*Au.trace(axis1=-2, axis2=-1)[..., np.newaxis]
        return val
        

    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:2], axis=-1)
        aphi[..., 0:2] -= lam/(2*mu+2*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi

    def div_stress(self, p):
        return -self.source(p)

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

