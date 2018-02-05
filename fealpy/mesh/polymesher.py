import numpy as np

def drectangle(p, box):
    return -dmin(
            dmin(dmin(p[...,1] - box[2], box[3]-p[:,1]), p[:,0] - box[0]),
            box[1] - p[:,0])  

class DistDomain2d:
    def __init__(self):
        pass

class RectangleDomain(DistDomain2d):
    def __init__(self):
        pass

    def distance(p):
        pass

    def __call__(p):
        pass


class PolyMesher:
    def __init__(self):
        pass

    
