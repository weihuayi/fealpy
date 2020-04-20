#!/usr/bin/env python3
# 
import numpy as np

class StructureMeshND:
    def __init__(self, box, N):
        self.box = box
        self.N = N
        self.GD = len(box)//2

        self.ftype = np.float
        self.itype = np.int32

    @property
    def node(self, flat=False):
        N = self.N
        GD = self.GD
        box = self.box
        node = np.mgrid[
                tuple(
                    slice(start, stop, complex(0, N+1))
                    for start, stop in zip(box[0::2], box[1::2])
                    )
                ]

        if flat:
            return node.reshape(-1).reshape(((N+1)*(N+1), GD), order='F')
        else:
            return node

    def ps_coefficients(self, flat=False):
        N = self.N
        GD = self.GD
        box = self.box
        n = N//2
        k = np.mgrid[tuple(slice(-n, n+1) for i in range(GD))]
        if flat:
            return k.reshape(-1).reshape(((N+1)*(N+1), GD), order='F')
        else:
            return k

    def ps_coefficients_square(self, flat=False):
        k = self.ps_coefficients(flat=flat)
        return np.sum(k**2, axis=0)
