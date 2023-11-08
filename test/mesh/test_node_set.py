import numpy as np
from fealpy.mesh import NodeSet 
import pytest

def test_init():
    import matplotlib.pyplot as plt
    mesh = NodeSet.from_dam_break_domain()
 
    fig, axes = plt.subplots()
    color = np.where(mesh.is_boundary_node, 'red', 'blue')
    mesh.add_plot(axes,color=color)
    #plt.show()

    dtype = [("velocity", "float64", (2, )),
             ("rho", "float64"),
             ("mass", "float64"),
             ("pressure", "float64"),
             ("sound", "float64")]
    mesh.add_node_data(dtype) 
    print(mesh.node_data("pressure"))

def test_neighbors():
    pass

def test_node_data():

    mesh = NodeSet.from_dam_break_domain()
    mesh.add_node_data(dtype)
if __name__ == "__main__":
    test_init()
    #test_node_data()

