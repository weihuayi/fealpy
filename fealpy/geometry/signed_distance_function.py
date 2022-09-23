import numpy as np
from .geoalg import project

class DistDomain2d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

class DistDomain3d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

def dcircle(p, cxy=[0, 0], r=1):
    x = p[..., 0]
    y = p[..., 1]
    return np.sqrt((x - cxy[0])**2 + (y - cxy[1])**2) - r

def drectangle(p, box):
    return -dmin(dmin(dmin(p[:, 1] - box[2], box[3]-p[:,1]), p[:,0] - box[0]), box[1] - p[:,0])

def dsine(p, cxy, r):
    x = p[:,0]
    y = p[:,1]
    return (y - cxy[1]) - r*np.sin(x-cxy[0])

def dparabolic(p, cxy, r):
    x = p[:,0]
    y = p[:,1]
    return (y - cxy[1])**2 - 2*r*x

def dcurve(p, curve, maxit=200, tol=1e-12):
    """

    Notes
    -----
    输入一组点和一个水平集函数表示的曲线，计算这些点到水平集的符号距离
    """
    _, d, _= project(curve, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
    return d

def dpoly(p, poly):
    pass

def ddiff(d0, d1):
    return np.maximum(d0, -d1)

#def dunion(d0, d1):
#    return np.minimum(d0, d1)

def dunion(*args):
    d = np.array(args)
    return np.min(d, axis=0)

#def dmin(d0, d1):
#    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
#    return dd.min(axis=1)

def dmin(*args):
    d = np.array(args)
    return np.min(d, axis=0)

#def dmax(d0, d1):
#    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
#    return dd.max(axis=1)

def dmax(*args):
    d = np.array(args)
    return np.max(d, axis=0)


def dcuboid(p, domain=[0, 1, 0, 1, 0, 1]):
    """
    @brief 长方体上的符号距离函数
    """
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]
    d = -dmin(
            z - domain[4], domain[5] - z, 
            y - domain[2], domain[3] - y, 
            x - domain[0], domain[1] - x)

    # (0, 1)
    val0 = domain[2] - y
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (1, 2)
    val0 = x - domain[1] 
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (2, 3)
    val0 = y - domain[3] 
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (0, 3)
    val0 = domain[0] - x
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (0, 4)
    val0 = domain[0] - x
    val1 = domain[2] - y 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (1, 5)
    val0 = x - domain[1]
    val1 = domain[2] - y 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (2, 6)
    val0 = x - domain[1]
    val1 = y - domain[3] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (3, 7)
    val0 = domain[0] - x
    val1 = y - domain[3] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (4, 5)
    val0 = domain[2] - y 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (5, 6)
    val0 = x - domain[1] 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (6, 7)
    val0 = y - domain[3] 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    # (4, 7)
    val0 = domain[0] - x 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

    return d

def dcylinder(p, 
        center=np.array([0.0, 0.0, 0.0]), 
        height=2, 
        radius=1, 
        direction=np.array([0.0, 0.0, 1.0])):
    """
    @brief 圆柱体的符号距离函数

    @param[in] p numpy 数组
    @param[in] c 圆柱体中心
    @param[in] h 圆柱体高度
    @param[in] r 圆柱体半径
    @param[in] d 圆柱体方向
    """

    v = p - center 
    d = np.sum(v*direction, axis=-1) # 中轴方向上到中心点的距离
    v -= d[..., None]*direction # v 到中轴的距离

    shape = p.shape[:-1] + (3, )
    val = np.zeros(shape, dtype=p.dtype)
    val[..., 0] = np.sqrt(np.sum(v**2, axis=1)) - radius # 到圆柱面的距离
    val[..., 1] =  d - height/2 # 到圆柱上圆面的距离
    val[..., 2] = -d - height/2 # 到圆柱下圆面的距离

    d = np.max(val, axis=-1)

    flag = (val[..., 0] > 0) & (val[..., 1] > 0)
    d[flag] = np.sqrt(val[flag, 0]**2 + val[flag, 1]**2)
    flag = (val[..., 0] > 0) & (val[..., 2] > 0)
    d[flag] = np.sqrt(val[flag, 0]**2 + val[flag, 2]**2)
    return d

def dsphere(p, 
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0):
    return np.sqrt(np.sum((p-center)**2, axis=-1)) - radius 
