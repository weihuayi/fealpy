import ipdb
import pytest
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt

from jax import random
bm.set_backend('jax')

def test_number_of_node():
    nodes = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    node_mesh =  NodeMesh(nodes)
    num = node_mesh.number_of_nodes()
    assert num == 3

def test_geo_dimension():
    nodes = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    node_mesh =  NodeMesh(nodes)
    geo_dim = node_mesh.geo_dimension()
    assert geo_dim == 2

def test_top_dimension():
    nodes = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    node_mesh =  NodeMesh(nodes)
    top = node_mesh.top_dimension()
    assert top == 0

def test_neighbors():
    box_size = 1.0  
    cutoff = 0.2
    key = random.PRNGKey(0)
    num_particles = 10
    positions = random.uniform(key, (num_particles, 2), minval=0.0, maxval=box_size)
    node_mesh = NodeMesh(positions)

    index, indptr = node_mesh.neighbors(box_size, cutoff)

    fig, ax = plt.subplots()
    node_mesh.add_plot(ax, color='red', markersize=25)
    for i in range(num_particles):
        ax.text(positions[i, 0], positions[i, 1], str(i), fontsize=12, ha='right')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet')
    plt.show()

def test_from_tgv_domain():
    box_size = bm.tensor([1.0,1.0])
    node = NodeMesh.from_tgv_domain(box_size)
    fig, ax = plt.subplots()
    node.add_plot(ax, color='red', markersize=25)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from TGV Domain')
    plt.show()
    
def test_from_heat_transfer_domain():
    nodemesh = NodeMesh.from_heat_transfer_domain()
    node = nodemesh.node
    color_map = {
        0: 'red',    # 流体节点为红色
        1: 'blue',   # 固体节点为蓝色
        3: 'green'   # 温度节点为绿色
    }
    fig, ax = plt.subplots()
    tag = nodemesh.nodedata["tag"]
    for t in bm.unique(tag):
        t_int = int(t.item())  
        idx = tag == t
        ax.scatter(node[idx, 0], node[idx, 1], color=color_map[t_int], s=25, label=f'Tag {t_int}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('NodeSet from dam break Domain')
    ax.legend()  
    plt.show()

def test_from_four_heat_transfer_domain():
    nodemesh = NodeMesh.from_four_heat_transfer_domain()
    node = nodemesh.node
    color_map = {
        0: 'red',    # 流体节点为红色
        1: 'blue',   # 固体节点为蓝色
        3: 'green'   # 温度节点为绿色
    }
    fig, ax = plt.subplots()
    tag = nodemesh.nodedata["tag"]
    for t in bm.unique(tag):
        t_int = int(t.item())  
        idx = tag == t
        ax.scatter(node[idx, 0], node[idx, 1], color=color_map[t_int], s=25, label=f'Tag {t_int}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('NodeSet from dam break Domain')
    ax.legend()  
    plt.show()

def test_from_long_rectangular_cavity_domain():
    uin = bm.tensor([5.0, 0.0])
    domain=[0,0.05,0,0.005] 
    init_domain=[0.0,0.005,0,0.005] 

    node_set = NodeMesh.from_long_rectangular_cavity_domain(init_domain=init_domain, domain=domain, uin=uin)
    wall_particles = node_set.nodedata["position"][node_set.nodedata["tag"] == 1]
    dummy_particles = node_set.nodedata["position"][node_set.nodedata["tag"] == 2]
    fuild_particles = node_set.nodedata["position"][node_set.nodedata["tag"] == 0]
    gate_particles = node_set.nodedata["position"][node_set.nodedata["tag"] == 3]
    
    fig, ax = plt.subplots()
    ax.scatter(wall_particles[:,0], wall_particles[:,1], color='red', s=25, label='wall_particles')
    ax.scatter(dummy_particles[:,0], dummy_particles[:,1], color='blue', s=25, label='dummy_particles')
    ax.scatter(fuild_particles[:,0], fuild_particles[:,1], color='orange', s=25, label='fuild_particles')
    ax.scatter(gate_particles[:,0], gate_particles[:,1], color='black', s=25, label='gate_particles')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from Domain')
    plt.show()

def test_from_slip_stick_domain():
    nodemesh = NodeMesh.from_slip_stick_domain()
    node = nodemesh.node
    color_map = {
        0: 'red',    # 流体节点为红色
        1: 'blue',   # 固体节点为蓝色
        2: 'orange', #移动固体节点为橙色
        3: 'green'   # 速度节点为绿色
    }
    fig, ax = plt.subplots()
    tag = nodemesh.nodedata["tag"]
    for t in bm.unique(tag):
        t_int = int(t.item())  
        idx = tag == t
        ax.scatter(node[idx, 0], node[idx, 1], color=color_map[t_int], s=25, label=f'Tag {t_int}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('NodeSet from dam break Domain')
    ax.legend()  
    plt.show() 

def test_from_dam_break_domain():
    node_set = NodeMesh.from_dam_break_domain()
    fig, ax = plt.subplots()
    node_set.add_plot(ax, color='red', markersize=50)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from dam break Domain')
    plt.show()

if __name__ == "__main__":
    test_number_of_node()
    test_geo_dimension()
    test_top_dimension()
    test_add_node_data()
    test_set_node_data()
    test_neighbors()
    test_from_tgv_domain()
    test_from_heat_transfer_domain()
    test_from_four_heat_transfer_domain()
    test_from_long_rectangular_cavity_domain()
    test_from_slip_stick_domain()
    test_from_dam_break_domain()