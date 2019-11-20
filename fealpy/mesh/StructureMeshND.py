import numpy as np


class StructureMeshND:
    def __init__(self, box, N):
        self.box = box
        self.N = N
        self.GD = len(box)//2

        self.ftype = type(box[0])
        self.itype = np.int

    @property
    def node(self):
        N = self.N
        GD = self.GD
        box = self.box
        node = np.mgrid[
                tuple(
                    slice(start, stop, complex(0, N+1))
                    for start, stop in zip(box[0::2], box[1::2])
                    )
                ]
        return node

    def ps_coefficients(self):
        N = self.N
        GD = self.GD
        box = self.box
        n = N//2
        k = np.mgrid[tuple(slice(-n, n+1) for i in range(GD))]
        return node

    def ps_coefficients_square(self):
        k = self.ps_coefficients()
        return np.sum(k**2, axis=0)
