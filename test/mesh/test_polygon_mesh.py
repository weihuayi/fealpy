import numpy as np
import ipdb
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh.polygon_mesh import PolygonMesh 

def test_polygon_mesh_constructor():
    node = np.array([
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
        (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    plt.show()


if __name__ == "__main__":
    test_polygon_mesh_constructor()

