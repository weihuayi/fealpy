
from ..backend import backend_manager as bm

def dcircle(p, cxy=[0, 0], r=1):
    x = p[..., 0]
    y = p[..., 1]
    return bm.sqrt((x - cxy[0])**2 + (y - cxy[1])**2) - r

def drectangle(p, box):
    x = p[..., 0]
    y = p[..., 1]
    d = dmin(y - box[2], box[3] - y)
    d = dmin(d, x - box[0])
    d = dmin(d, box[1] - x)
    return -d

def dsine(p, cxy, r):
    x = p[..., 0]
    y = p[..., 1]
    return (y - cxy[1]) - r*bm.sin(x-cxy[0])

def dparabolic(p, cxy, r):
    x = p[..., 0]
    y = p[..., 1]
    return (y - cxy[1])**2 - 2*r*x

def ddiff(d0, d1):
    return bm.maximum(d0, -d1)

def dunion(*args):
    d = bm.array(args)
    return bm.min(d, axis=0)

def dmin(*args):
    d = bm.array(args)
    return bm.min(d, axis=0)

def dmax(*args):
    d = bm.array(args)
    return bm.max(d, axis=0)

def dintersection(d0, d1):
    return bm.maximum(d0, d1)

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
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (1, 2)
    val0 = x - domain[1] 
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (2, 3)
    val0 = y - domain[3] 
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (0, 3)
    val0 = domain[0] - x
    val1 = domain[4] - z
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (0, 4)
    val0 = domain[0] - x
    val1 = domain[2] - y 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (1, 5)
    val0 = x - domain[1]
    val1 = domain[2] - y 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (2, 6)
    val0 = x - domain[1]
    val1 = y - domain[3] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (3, 7)
    val0 = domain[0] - x
    val1 = y - domain[3] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (4, 5)
    val0 = domain[2] - y 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (5, 6)
    val0 = x - domain[1] 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (6, 7)
    val0 = y - domain[3] 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    # (4, 7)
    val0 = domain[0] - x 
    val1 = z - domain[5] 
    flag = (val0 > 0) & (val1 > 0)
    d[flag] = bm.sqrt(val0[flag]**2 + val1[flag]**2)

    return d

def dcylinder(p, 
        center=bm.array([0.0, 0.0, 0.0]), 
        height=2, 
        radius=1, 
        direction=bm.array([0.0, 0.0, 1.0])):
    """
    @brief 圆柱体的符号距离函数

    @param[in] p numpy 数组
    @param[in] c 圆柱体中心
    @param[in] h 圆柱体高度
    @param[in] r 圆柱体半径
    @param[in] d 圆柱体方向
    """

    v = p - center 
    d = bm.sum(v*direction, axis=-1) # 中轴方向上到中心点的距离
    v -= d[..., None]*direction # v 到中轴的距离

    shape = p.shape[:-1] + (3, )
    val = bm.zeros(shape, dtype=p.dtype)
    val[..., 0] = bm.sqrt(bm.sum(v**2, axis=1)) - radius # 到圆柱面的距离
    val[..., 1] =  d - height/2 # 到圆柱上圆面的距离
    val[..., 2] = -d - height/2 # 到圆柱下圆面的距离

    d = bm.max(val, axis=-1)

    flag = (val[..., 0] > 0) & (val[..., 1] > 0)
    d[flag] = bm.sqrt(val[flag, 0]**2 + val[flag, 1]**2)
    flag = (val[..., 0] > 0) & (val[..., 2] > 0)
    d[flag] = bm.sqrt(val[flag, 0]**2 + val[flag, 2]**2)
    return d

def dsphere(p, 
        center=bm.array([0.0, 0.0, 0.0]),
        radius=1.0):
    return bm.sqrt(bm.sum((p-center)**2, axis=-1)) - radius 
