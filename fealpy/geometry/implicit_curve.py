import numpy as np
from .geoalg import project


class PolygonCurve():
    """

    Notes
    -----
    用于描述一个多边形区域的边界。目前假设区域内部没有内部边界，即区域是连通的。

    TODO
    ----
    1. 考虑区域有内部边界的情形
    """
    def __init__(self, node, edge, edge2subdomain=None):
        self.node = node
        self.edge = edge
        self.edge2subdomain = edge2subdomain

    def __call__(self, p):
        """

        Notes
        -----
        给定点集 p， 计算它们到多边形边界的距离。
        """

        node = self.node
        edge = self.edge

        NE = len(edge)
        NP = len(p) # 这里假设 p 是二维数组 TODO：考虑只有一个点的情形
        v = node[edge[:, 1]] - node[edge[:, 2]]
        l = np.sqrt(np.sum(v**2, axis=1))
        v /= l.reshape(-1, 1)
        w = np.array([(0,-1),(1,0)])
        n = v@w

        # 计算符号距离
        # (NP, 2) - (NE, 2)->(NP, NE, 2), (NE, 2) -> (NP, NE)  
        d = np.sum((p[..., None, :] - node[edge[:, 0]])*n, axis=-1)
        return d

class CircleCurve():
    def __init__(self, center=np.array([0.0, 0.0]), radius=1.0):
        self.center = center
        self.radius = radius
        self.box = [-1.5, 1.5, -1.5, 1.5]

    def init_mesh(self, n):
        from ..mesh import IntervalMesh

        dt = 2*np.pi/n
        theta  = np.arange(0, 2*np.pi, dt)

        node = np.zeros((n, 2), dtype = np.float64)
        cell = np.zeros((n, 2), dtype = np.int_)

        node[:, 0] = self.radius*np.cos(theta)
        node[:, 1] = self.radius*np.sin(theta)
        node += self.center

        cell[:, 0] = np.arange(n)
        cell[:, 1][:-1] = np.arange(1,n)

        mesh = IntervalMesh(node, cell)

        return mesh 

    def __call__(self, p):
        return np.sqrt(np.sum((p - self.center)**2, axis=-1))-self.radius

    def value(self, p):
        return self(p)

    def gradient(self, p):
        l = np.sqrt(np.sum((p - self.center)**2, axis=-1))
        n = (p - self.center)/l[..., np.newaxis]
        return n

    def distvalue(self, p):
        p, d, n= project(self, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
        return d, n

    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d 

class DoubleCircleCurve():
    """
    @brief 两个圆相交的曲线
    """
    def __init__(self, l = 1.0, radius=np.sqrt(2), center=np.array([0, 0])):
        self.circle0 = CircleCurve(center-np.array([l, 0]), radius)
        self.circle1 = CircleCurve(center+np.array([l, 0]), radius)
        self.center = center
        self.b = np.sqrt(radius**2 - l**2)
        self.l = l
        self.k = self.b/l
        self.box = [-3, 3, -2, 2]

    def init_mesh(self, n):
        pass

    def __call__(self, p):
        k = self.k
        b = self.b
        l = self.l

        x = p[..., 0] - self.center[0]
        y = p[..., 1] - self.center[1]
        yflag = y>0
        flag0 = yflag & (y<b) & (x < -y/k + l) &(x > y/k-l)
        flag1 = (~yflag) & (y>-b) & (x < y/k + l) &(x > -y/k-l)

        val = np.zeros(p.shape[:-1], dtype=np.float64)
        val[flag0] = -np.sqrt(x[flag0]**2 + (y[flag0]-b)**2)
        val[flag1] = -np.sqrt(x[flag1]**2 + (y[flag1]+b)**2)

        xflag = x>0
        flag23 = ~(flag0|flag1)
        flag2 = xflag & flag23
        flag3 = (~xflag) & flag23
        val[flag2] = self.circle1(p[flag2])
        val[flag3] = self.circle0(p[flag3])
        return val 

    def value(self, p):
        return self(p)

    def gradient(self, p):
        k = self.k
        b = self.b
        l = self.l

        val = self(p)

        x = p[..., 0] - self.center[0]
        y = p[..., 1] - self.center[1]
        yflag = y>0
        flag0 = yflag & (y<b) & (x < -y/k + l) &(x > y/k-l) # Omega0
        flag1 = (~yflag) & (y>-b) & (x < y/k + l) &(x > -y/k-l) # Omega1

        xflag = x>0
        flag23 = ~(flag0|flag1)
        flag2 = xflag & flag23 # Omega2
        flag3 = (~xflag) & flag23 # Omega3

        gval = np.zeros(p.shape, dtype=np.float64)
        gval[flag0, 0] = x[flag0]/val[flag0] 
        gval[flag1, 0] = x[flag1]/val[flag1] 
        gval[flag2, 0] = (x[flag2]-l)/val[flag2] 
        gval[flag3, 0] = (x[flag3]+l)/val[flag3] 
        return gval

    def distvalue(self, p):
        p, d, n= project(self, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
        return d, n

    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d 

class DoubleBandY():
    def __init__(self, ylist=[0.24, 0.26, 0.74, 0.76]):
        self.ylist = ylist
        self.cen = [0.5*(ylist[0]+ylist[1]), 0.25*sum(ylist), 0.5*(ylist[2]+ylist[3])]

    def init_mesh(self, n):
        pass

    def __call__(self, p):
        x = p[..., 0]
        y = p[..., 1]

        flag0 = y < self.cen[0]
        flag1 = (~flag0) & (y<self.cen[1])
        flag2 = (~flag0) & (~flag1) & (y<self.cen[2])
        flag3 = (~flag0) & (~flag1) & (~flag2)

        val = np.zeros(p.shape[:-1], dtype=np.float64)
        val[flag0] = self.ylist[0]-y[flag0]
        val[flag1] = y[flag1]-self.ylist[1] 
        val[flag2] = self.ylist[2]-y[flag2]
        val[flag3] = y[flag3]-self.ylist[3] 
        return val 

    def value(self, p):
        return self(p)

    def gradient(self, p):
        k = self.k
        b = self.b
        l = self.l

        val = self(p)

        x = p[..., 0] - self.center[0]
        y = p[..., 1] - self.center[1]
        yflag = y>0
        flag0 = yflag & (y<b) & (x < -y/k + l) &(x > y/k-l) # Omega0
        flag1 = (~yflag) & (y>-b) & (x < y/k + l) &(x > -y/k-l) # Omega1

        xflag = x>0
        flag23 = ~(flag0|flag1)
        flag2 = xflag & flag23 # Omega2
        flag3 = (~xflag) & flag23 # Omega3

        gval = np.zeros(p.shape, dtype=np.float64)
        gval[flag0, 0] = x[flag0]/val[flag0] 
        gval[flag1, 0] = x[flag1]/val[flag1] 
        gval[flag2, 0] = (x[flag2]-l)/val[flag2] 
        gval[flag3, 0] = (x[flag3]+l)/val[flag3] 

        x = p[..., 0]
        y = p[..., 1]

        flag0 = y < self.cen[0]
        flag1 = (~flag0) & (y<self.cen1[1])
        flag2 = (~flag0) & (~flag1) & (y<self.cen1[2])
        flag3 = (~flag0) & (~flag1) & (~flag2)

        gval = np.zeros(p.shape, dtype=np.float64)
        gval[flag0, 1] = -1
        gval[flag1, 1] =  1
        gval[flag2, 1] = -1
        gval[flag3, 1] =  1
        return gval

    def distvalue(self, p):
        p, d, n= project(self, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
        return d, n

    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d 

class FoldCurve():
    def __init__(self, a=6):
        self.a = a
        self.box = [-1, 1, -1, 1]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        c = 0.02*np.sqrt(5)
        pi = np.pi
        theta = np.arctan2(y- c, x - c)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)

        return (x - c)**2 + (y - c)**2 - (0.5 + 0.2 * np.sin(a*theta))**2 

    def value(self, p):
        return self(p)

    def gradient(self, p):
        c = 0.02*np.sqrt(5)
        a = self.a
        x = p[..., 0]
        y = p[..., 1]
        b0 = 0.5
        b1 = 0.2
        cos = np.cos
        sin = np.sin
        pi = np.pi

        theta = np.arctan2(y- c, x - c)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)
        r2 = (-c + x)**2 + (-c + y)**2

        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -2*a*b1*(b0 + b1*sin(a*theta))*(c - y)*cos(a*theta)/r2 - 2*c + 2*x

        grad[..., 1] = -2*a*b1*(b0 + b1*sin(a*theta))*(-c + x)*cos(a*theta)/r2 - 2*c + 2*y
        return grad

    def distvalue(self, p):
        p, d, n= project(self, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
        return d, n

    def project(self, p):
        p, d, n= project(self, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
        return p, d


class Curve2():
    def __init__(self, r0=0.60125, r1=0.24012):
        self.r0 = r0
        self.r1 = r1
        self.box = [-1, 1, -1, 1]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        r0 = self.r0
        r1 = self.r1
        pi = np.pi

        x1 = np.arccos(-1/4)/4
        x0 = pi/2 - x1 - np.sin(4*x1)
 
        theta = np.arctan2(y, x)
        isNeg = theta < 0
        theta[isNeg] = theta[isNeg] + 2*pi

        z = np.zeros(len(x), dtype=np.float)
        rp = np.sqrt(x**2 + y**2)

        isSingle0 = (theta >= 0) & (theta < x0) # [0,x0]
        isSingle1 = (theta > pi/2 - x0) & (theta < pi/2 + x0) # [pi/2 - x0, pi/2 + x0]
        isSingle2 = (theta > pi - x0) & (theta < pi + x0) # [pi - x0, pi + x0]
        isSingle3 = (theta > 3*pi/2 - x0) & (theta < 3*pi/2 +x0) # [3*pi/2 - x0, 3*pi/2 + x0]
        isSingle4 = (theta > 2*pi - x0) & (theta <= 2*pi) # [2*pi - x0, 2*pi]

        isThree0 = (theta >= x0) & (theta <= pi/2 - x0) # [x0,x1], [x1,pi/2-x1],[pi/2 - x1,pi/2-x0]
        isThree1 = (theta >= pi/2 + x0) & (theta <= pi - x0) # [pi/2 + x0, pi/2 + x1], [ pi/2 + x1, pi - x1],[pi - x1, pi - x0]
        isThree2 = (theta >= pi + x0) & (theta <= 3*pi/2 - x0) # [pi + x0, pi + x1],[pi + x1, 3*pi/2 - x1],[3*pi/2 - x1,3*pi/2 - x0]
        isThree3 = (theta >= 3*pi/2 + x0) & (theta <= 2*pi - x0) # [3*pi/2 + x0, 3*pi/2 + x1], [ 3*pi/2 + x1, 2*pi - x1],[2*pi - x1, 2*pi - x0]

        isSingle = (isSingle0 | isSingle1 | isSingle2 | isSingle3 | isSingle4)
        theta1 = theta[isSingle]
        t0 = np.zeros(len(x), dtype=np.float)
        t0[isSingle0] = x0/2
        t0[isSingle1] = pi/2
        t0[isSingle2] = pi
        t0[isSingle3] = 3*pi/2
        t0[isSingle4] = 2*pi - x0/2

        if len(theta1)>0:
           t = self.get_T(theta1.reshape(-1, 1), t0[isSingle])
           r = r0 + r1*np.cos(4*t + pi/2)
           z[isSingle] = rp[isSingle]**2 - r**2

        isThree = (isThree0 | isThree1 | isThree2 | isThree3)
        theta1 = theta[isThree]
        z1 = np.zeros(len(theta1), dtype=np.float)

        t0[isThree0, 0] = (x0 + x1)/2
        t0[isThree0, 1] = pi/4
        t0[isThree0, 2] = pi/2 - (x0 + x1)/2

        t0[isThree1, 0] = pi/2 + (x0 + x1)/2
        t0[isThree1, 1] = 3*pi/4
        t0[isThree1, 2] = pi - (x0 + x1)/2

        t0[isThree2, 0] = pi + (x0 + x1)/2
        t0[isThree2, 1] = 5*pi/4
        t0[isThree2, 2] = 3*pi/2 - (x0 + x1)/2

        t0[isThree3, 0] = 3*pi/2 + (x0 + x1)/2
        t0[isThree3, 1] = 7*pi/4
        t0[isThree3, 2] = 2*pi - (x0 + x1)/2

        if np.any(isThree):
           t = self.get_T(theta1.reshape(-1, 1), t0[isThree])
           r = r0 + r1*np.cos(4*t + pi/2) 
           rt = rp[isThree]
           flag1 = (rt < (r[:,0] + r[:,1])/2)
           flag2 = (rt >= (r[:,0] + r[:,1])/2) & (rt < (r[:, 1] + r[:,2])/2)
           flag3 = (rt >= (r[:,1] + r[:,2])/2)
           if np.any(flag1):
               z1[flag1] = rt[flag1]**2 - r[flag1, 0]**2

           if np.any(flag2):
               z1[flag2] = r[flag2, 1]**2 - rt[flag2]**2

           if np.any(flag3):
               z1[flag3] = rt[flag3]**2- r[flag3, 2]**2

           z[isThree] = z1

        tt = np.array([x1, 2*pi - x1, pi/2 - x1, pi/2 + x1, pi - x1, pi + x1, 3*pi/2 - x1, 3*pi/2 + x1], dtype=np.float)
        rt = r0 + r1 * np.cos(4*tt + pi/2);
        xt = rt*np.cos( tt + np.sin(4*tt));
        yt = rt*np.sin( tt + np.sin(4*tt));
        rt = np.zeros((len(x), 8), dtype=np.float)
        for i in range(8): 
            rt[:,i] = np.sqrt((x - xt[i])**2 + (y - yt[i])**2)    

        rt = np.min(rt, axis=1)

        s = np.sign(z)
        u = np.abs(z)
        isBigger = u > rt

        u[isBigger] = rt[isBigger]
        return s*u

    def get_T(self, theta, t0):
        eps = 1e-8
        f = t0 + np.sin(4*t0) - theta
        fprime = 1 + 4*np.cos(4*t0)
        t = t0 - f/fprime
        err = np.sqrt(sum(sum(f**2)))
        while err > eps:
            t0 = t
            f = t0 + np.sin(4*t0) - theta
            fprime = 1 + 4*np.cos(4*t0)
            t = t0 - f/fprime
            err = np.sqrt(np.sum(f**2))
    
class Curve3():

    def __init__(self):
        self.box = [-25, 25, -25, 25]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        r = x**2 + y**2
        theta = np.arctan2(y,x)
        isNeg = theta < 0
        theta[isNeg] = theta[isNeg] + 2*np.pi
        x1 = 16*np.sin(theta)**3
        y1 = 13*np.cos(theta) - 5*np.cos(2*theta) - 2*np.cos(3*theta) - np.cos(4*theta)
        return r - (x1**2 + y1**2)

class BicornCurve():
    '''
    Bicorn curve
    '''
    def __init__(self, a):
        self.a = a

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a

        return y**2*(a**2 - x**2) - (x**2 + 2*a*y - a**2)**2

class CardioidCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/Cardioid.html
        r = 2*a*(1 + cos(theta))
    '''
    def __init__(self, a):
        self.a = a

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        r2 = x**2 + y**2

        return (r2- 2*a*x)**2 - 4*a**2*r2

class CartesianOvalCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/Cartesian.html
    '''
    def __init__(self, a, c, m):
        self.a = a
        self.c = c
        self.m = m

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        c = self.c
        m = self.m
        r2 = x**2 + y**2
        l = (1-m**2)*r2 + 2*m**2*c*x + a**2 - m**2*c**2

        return l**2 - 4*a**2*r2 

class CassinianOvalsCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/Cassinian.html
    '''
    def __init__(self, a, c):
        self.a = a
        self.c = c

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        c = self.c
        r2 = x**2 + y**2
        m2 = x**2 - y**2

        return r2**2 - 2*a**2*m2 + a**4 - c**2 

class FoliumCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/Folium.html
        r = -b*cos(theta) + 4*a*cos(theta)*sin^2(theta)
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        b = self.b
        r2 = x**2 + y**2

        return r2*(r2 + x*b) - 4*a*x*y**2

class LameCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/Lame.html
        r = -b*cos(theta) + 4*a*cos(theta)*sin^2(theta)
    '''
    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        b = self.b
        n = self.n

        return (x/a)**n + (y/b)**n - 1 

class PearShapedCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/PearShaped.html
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        b = self.b

        return  b**2*y**2 - x**3*(a - x)

class SpiricSectionsCurve():
    '''
        http://www-gap.dcs.st-and.ac.uk/~history/Curves/Spiric.html
    '''
    def __init__(self, a, c, r):
        self.a = a
        self.c = c
        self.r = r

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError("the args must be a N*2 or (X, Y)")

        a = self.a
        c = self.c
        r = self.r

        return (r**2 - a**2 + c**2 + x**2 + y**2)**2 - 4*r**2*(x**2 + c**2)
        

        return  b**2*y**2 - x**3*(a - x)
