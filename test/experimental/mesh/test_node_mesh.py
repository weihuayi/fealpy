from fealpy.experimental.mesh.node_mesh import NodeMesh
from fealpy.experimental.backend import backend_manager as bm
import matplotlib.pyplot as plt
bm.set_backend('jax')

def test_number_of_node():
    nodes = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    node_mesh =  NodeMesh(nodes)
    num = node_mesh.number_of_nodes()

def test_geo_dimension():
    nodes = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    node_mesh =  NodeMesh(nodes)
    geo_dim = node_mesh.geo_dimension()

def test_top_dimension():
    nodes = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    node_mesh =  NodeMesh(nodes)
    top = node_mesh.top_dimension()

def test_add_node_data():
    nodes = bm.tensor([[0, 0], [1, 1], [2, 2]], dtype=bm.float32)
    node_mesh = NodeMesh(nodes)

    temperature_data = bm.tensor([300.0, 350.0, 400.0], dtype=bm.float32)
    pressure_data = bm.tensor([100.0, 200.0, 300.0], dtype=bm.float32)
    node_mesh.add_node_data(['temperature', 'pressure'], [temperature_data, pressure_data])

def test_set_node_data():
    nodes = bm.tensor([[0, 0], [1, 1], [2, 2]], dtype=bm.float32)
    node_mesh = NodeMesh(nodes)

    temperature_data = bm.tensor([300.0, 350.0, 400.0], dtype=bm.float32)
    pressure_data = bm.tensor([100.0, 200.0, 300.0], dtype=bm.float32)
    node_mesh.add_node_data(['temperature', 'pressure'], [temperature_data, pressure_data])
    new_tem = bm.tensor([100,100,100], dtype=bm.float32)
    node_mesh.set_node_data('temperature', new_tem)

def test_from_tgv_domain():
    box_size = bm.tensor([1.0,1.0])
    node = NodeMesh.from_tgv_domain(box_size)
    fig, ax = plt.subplots()
    node.add_plot(ax, color='red', markersize=25)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from TGV Domain')
    plt.show()
    

if __name__ == "__main__":
    test_number_of_node()
    test_geo_dimension()
    test_top_dimension()
    test_add_node_data()
    test_set_node_data()
    test_from_tgv_domain()

    