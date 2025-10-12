from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh


if __name__ == "__main__":
    l_shape_mesh = TetrahedronMesh.from_box()