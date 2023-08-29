import numpy as np

from ..decorator import cartesian 
from ..mesh import EdgeMesh


class BeamBase:
    @staticmethod
    def convert_units(length_in_inches, force_in_pounds):
        """
        英寸 in 转换成毫米 mm 
        磅 lb 转换成牛顿 newton
        """
        length_in_mm = length_in_inches * 25.4
        force_in_newtons = force_in_pounds * 4.44822
        return length_in_mm, force_in_newtons


class Beam_2d_cantilever():
    def __init__(self):
        self.I = 118.6e-6 # 惯性矩 m^4
        self.E = 200e9 # 弹性模量 newton/m^2

    def init_mesh(self):
        mesh = EdgeMesh.from_cantilever()
        return mesh


class Beam_2d_three_beam():
    def __init__(self):
        self.I = 6.5e-7 # 惯性矩 m^4
        self.E = 3e11 # 弹性模量 ton/m^2

    def init_mesh(self):
        mesh = EdgeMesh.from_three_beam()
        return mesh
