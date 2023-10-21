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


class CantileveredSimplySupportedBeam():
    def __init__(self):
        self.I = 1.186e-4 # Moment of Inertia - m^4
        self.A = 6650e-6 # Cross-sectional area - m^2
        self.E = 200e9 # Elastic Modulus newton - m^2

    def init_mesh(self):
        mesh = EdgeMesh.generate_cantilevered_mesh()
        return mesh


class PlanarBeam():
    def __init__(self):
        self.I = 6.5e-7 # Moment of Inertia - m^4
        self.A = 6.8e-4 # Cross-sectional area - m^2
        self.E = 3e11 # Elastic Modulus-newton - m^2


    def init_mesh(self):
        mesh = EdgeMesh.generate_tri_beam_frame_mesh()
        return mesh


