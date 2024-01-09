import numpy as np
class ConvectionDiffusionModel:
    def __init__(self):
        pass
    def domain(self):
        return np.array([0,1,0,1])

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return x*y*0
    def d1(self, p):
        eps = 1e-10
        x = p[..., 0]
        y = p[..., 1]
        flag = ((np.abs(x)<eps)& (y>0.7-eps) & (y<1+eps) )| (np.abs(y-1)<eps)
        return flag
    def d2(self, p):
        eps = 1e-10
        x = p[..., 0]
        y = p[..., 1]
        flag = ((np.abs(x)<eps)& (y<0.7+eps) &(y>-eps)) | (np.abs(x-1)<eps)
        return flag
    def d(self, p):
        eps = 1e-10
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(y)<eps) | (np.abs(x-1)<eps)| (np.abs(x)<eps) | (np.abs(y-1)<eps)
        return flag


