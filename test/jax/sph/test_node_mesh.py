import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph.partition import *
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md.partition import space
import matplotlib.pyplot as plt


def test_neighbors():
    box_size = 1.0  
    cutoff = 0.2
    key = random.PRNGKey(0)
    num_particles = 10
    positions = random.uniform(key, (num_particles, 2), minval=0.0, maxval=box_size)
    node_mesh = NodeMesh(positions)
    print(node_mesh.number_of_node())
    # 计算邻近列表
    #index, indptr = node_mesh.neighbors(box_size, cutoff)
    #print(index)
    #print(indptr)
    
def test_neighbors_jax():
    box_size = 1.0  
    cutoff = 0.2
    key = random.PRNGKey(0)
    num_particles = 10
    positions = random.uniform(key, (num_particles, 2), minval=0.0, maxval=box_size)
    node_mesh = NodeMesh(positions)
    
    displacement_fn, shift_fn = space.periodic(side=box_size)
    neighbor_fn = neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=cutoff,
        mask_self=False
    )
    neighbors = neighbor_fn.allocate(positions)
    neighbors = neighbor_fn.allocate(positions)
    # 计算邻近列表
    #index, indptr = node_mesh.neighbors(box_size, cutoff)
    #print(index)
    print(neighbors.idx)

def test_add_node_data():
    #创建初始粒子
    nodes = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    node_set = NodeMesh(nodes)
    # 添加节点数据
    node_set.add_node_data(['temperature', 'pressure'], [jnp.float64, jnp.float32])
    # 设置节点数据
    temperature_data = jnp.array([300.0, 350.0, 400.0])
    pressure_data = jnp.array([100.0, 200.0, 300.0])
    node_set.set_node_data('temperature', temperature_data)
    node_set.set_node_data('pressure', pressure_data)
    # 检查节点数据是否正确
    assert 'temperature' in node_set.nodedata, "Temperature data not found"
    assert 'pressure' in node_set.nodedata, "Pressure data not found"
    assert jnp.array_equal(node_set.nodedata['temperature'], temperature_data), "Temperature data mismatch"
    assert jnp.array_equal(node_set.nodedata['pressure'], pressure_data), "Pressure data mismatch"
    print("all tests passed.")

def test_interpolate():
    #参数设置
    box_size = 10.0  
    cutoff = 1.5 
    key = random.PRNGKey(0)
    num_particles = 10
    positions = random.uniform(key, (num_particles, 2), minval=0.0, maxval=box_size)
    node_set = NodeMesh(positions)
    #测试函数
    u = node_set.node[:,0]*node_set.node[:,1] 
    kernel = QuinticKernel(h=cutoff,dim=2)
    neighbor = node_set.neighbors(box_size, cutoff)
    a = node_set.interpolate(u,kernel,neighbor,cutoff)
    print(a[0])
    print(a[1])

def test_from_tgv_domain():
    box_size = jnp.array([1.0,1.0])
    node = NodeMesh.from_tgv_domain(box_size)
    fig, ax = plt.subplots()
    node.add_plot(ax, color='red', markersize=25)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from TGV Domain')
    plt.show()

def test_from_ringshaped_channel_domain():
    node_set = NodeMesh.from_ringshaped_channel_domain()
    fig, ax = plt.subplots()
    node_set.add_plot(ax, color='red', markersize=50)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from ring shaped channel Domain')
    plt.show()

def test_dam_break_domain():
    node_set = NodeMesh.from_dam_break_domain()
    fig, ax = plt.subplots()
    node_set.add_plot(ax, color='red', markersize=50)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from dam break Domain')
    plt.show()

def test_long_rectangular_cavity_domain():
    node_set = NodeMesh.from_long_rectangular_cavity_domain()
    wall_particles = node_set.nodedata["position"][node_set.nodedata["tag"] == 1]
    dummy_particles = node_set.nodedata["position"][node_set.nodedata["tag"] == 2]
    fig, ax = plt.subplots()
    ax.scatter(wall_particles[:,0], wall_particles[:,1], color='red', s=25, label='wall_particles')
    ax.scatter(dummy_particles[:,0], dummy_particles[:,1], color='blue', s=25, label='dummy_particles')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('NodeSet from Domain')
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
    for t in np.unique(tag):
        idx = tag == t
        ax.scatter(node[idx, 0], node[idx, 1], color=color_map[t], s=25, label=f'Tag {t}')
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
    for t in np.unique(tag):
        idx = tag == t
        ax.scatter(node[idx, 0], node[idx, 1], color=color_map[t], s=25, label=f'Tag {t}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('NodeSet from dam break Domain')
    ax.legend()  
    plt.show()

def test_from_slip_stick_domain():
    nodemesh = NodeMesh.from_slip_stick_domain()
    node = nodemesh.node
    color_map = {
        0: 'red',    # 流体节点为红色
        1: 'blue',   # 固体节点为蓝色
        3: 'green'   # 速度节点为绿色
    }
    fig, ax = plt.subplots()
    tag = nodemesh.nodedata["tag"]
    for t in np.unique(tag):
        idx = tag == t
        ax.scatter(node[idx, 0], node[idx, 1], color=color_map[t], s=25, label=f'Tag {t}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('NodeSet from dam break Domain')
    ax.legend()  
    plt.show()

if __name__ == "__main__":
    #test_neighbors()
    #test_neighbors()
    #test_neighbors_jax()
    #test_add_node_data()
    #test_interpolate()
    #test_from_tgv_domain()
    #test_from_ringshaped_channel_domain()
    #test_dam_break_domain()
    #test_from_heat_transfer_domain()
    #test_from_four_heat_transfer_domain()
    #test_from_slip_stick_domain()
    test_long_rectangular_cavity_domain()