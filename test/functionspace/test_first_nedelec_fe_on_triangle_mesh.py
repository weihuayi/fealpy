import numpy as np
import matplotlib.pyplot as plt
import pytest
import ipdb

from fealpy.functionspace import FirstNedelecFEOnTriangleMesh 
from fealpy.mesh import TriangleMesh

@pytest.mark.parametrize("p", range(1, 10))
def test_dofs(p):
    mesh = TriangleMesh.from_one_triangle()
    
    space = FirstNedelecFEOnTriangleMesh(mesh, p)

    ips = space.interpolation_points()

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ips, showindex=True)
    plt.show()

if __name__ == '__main__':
    test_dofs(3)
