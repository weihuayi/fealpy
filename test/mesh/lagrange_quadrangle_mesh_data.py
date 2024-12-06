import numpy as np
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh

# 定义多个典型的 LagrangeTriangleMesh 对象
surface = SphereSurface() #以原点为球心，1 为半径的球
mesh = QuadrangleMesh.from_unit_sphere_surface()
node = mesh.interpolation_points(3)
cell = mesh.cell_to_ipoint(3)

cell_area_data = [
        {
            "sphere_cm": 4*np.pi*(1**2)
            }
]

