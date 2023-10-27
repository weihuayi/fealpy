import numpy as np
from fealpy.mesh import NodeSet 
import pytest

def test_init():
    import matplotlib.pyplot as plt
    mesh = NodeSet.from_dam_break_domain()

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    plt.show()

def test_neighbors():
    pass

def test_node_data():

    dtype = [("position", "float64", (2, )), 
             ("velocity", "float64", (2, )),
             ("rho", "float64"),
             ("mass", "float64"),
             ("pressure", "float64"),
             ("sound", "float64")]

    data = np.zeros(


if __name__ == "__main__":
    test_init()
