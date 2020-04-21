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
    def node(self):
        N = self.N
        GD = self.GD
        box = self.box
        node = np.ogrid[
                tuple(
                    slice(start, stop-(stop - start)/N, complex(0, N))
                    for start, stop in zip(box[0::2], box[1::2])
                    )
                ]

        return node

    def freq(self):
        N = self.N
        GD = self.GD
        xi = np.fft.fftfreq(N)*N
        xi = np.meshgrid(*(GD*(xi,)), sparse=True)
        return xi

    def linear_equation(self, f, cfun=lambda x: 1 + np.sum(x**2, axis=0)):
        N = self.N
        GD = self.GD
        xi = self.freq()
        F = f(self.node)[
        F = np.fft.fft(F)
        U = F/cfun(xi)
        print('U1:', U1)
        U1 = np.fft.ifft(U1).real



