import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.mesh.interval_mesh import IntervalMesh

def test_interval_domain():
    mesh = IntervalMesh.from_interval_domain([0, 1], nx=10)
    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True) 
    plt.show()



if __name__ == '__main__':
    test_interval_domain()

