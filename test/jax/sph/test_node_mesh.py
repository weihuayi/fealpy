import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from fealpy.jax.sph.node_mesh import NodeMesh
from fealpy.jax.sph.kernel_function import QuinticKernel
import matplotlib.pyplot as plt

def test_neighbor():
    box_size = 1.0
    h = 0.2
    key = random.PRNGKey(0)
    num_particles = 100
    positions = random.uniform(key, (num_particles, 2), minval=0.0, maxval=box_size)
    node_mesh = NodeMesh(positions)
    idx = node_mesh.neighbor(box_size,h)
    print(idx)

def test_neighbors():
    box_size = 1.0  
    cutoff = 0.2
    key = random.PRNGKey(0)
    num_particles = 10
    positions = random.uniform(key, (num_particles, 2), minval=0.0, maxval=box_size)
    node_mesh = NodeMesh(positions)
    
    # 计算邻近列表
    neighbors_dict = node_mesh.neighbors(box_size, cutoff)
    print(neighbors_dict)
    '''
    for i, data in neighbors_dict.items():
        print(f"Particle {i} neighbors: {data['indices']}")
        for j, distance in zip(data['indices'], data['distances']):
            print(f"  Distance between particle {i} and {j}: {distance}")
    '''
    
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
    node_fluid, node_dummy = NodeMesh.from_tgv_domain()
    fig, ax = plt.subplots()
    node_fluid.add_plot(ax, color='red', markersize=25)
    node_dummy.add_plot(ax,color='blue', markersize=25)
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

if __name__ == "__main__":
    #test_neighbor()
    #test_neighbors()
    #test_add_node_data()
    #test_interpolate()
    test_from_tgv_domain()
    #test_from_ringshaped_channel_domain()
    #test_dam_break_domain()