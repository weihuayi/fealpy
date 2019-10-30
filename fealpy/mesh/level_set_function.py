import numpy as np

class DistDomain2d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

class DistDomain3d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args


