import numpy as np
from .geoalg import project

class DistDomain2d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

class DistDomain3d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

def dcircle(p, cxy=[0, 0], r=1):
    x = p[:, 0]
    y = p[:, 1]
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
    return dmax(d0, -d1)

def dmin(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.min(axis=1)

def dmax(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.max(axis=1)
