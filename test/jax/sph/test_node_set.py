import jax.numpy as jnp
from jax import random
from fealpy.jax.sph.node_set import NodeSet

def test_neighbors():
    box_size = 10.0  
    cutoff = 1.0 
    key = random.PRNGKey(0)
    num_particles = 100
    positions = random.uniform(key, (num_particles, 3), minval=0.0, maxval=box_size)
    node_set = NodeSet(positions)
    
    # 计算邻近列表
    neighbors_dict = node_set.neighbors(box_size, cutoff)

    for i, data in neighbors_dict.items():
        print(f"Particle {i} neighbors: {data['indices']}")
        for j, distance in zip(data['indices'], data['distances']):
            print(f"  Distance between particle {i} and {j}: {distance}")

def test_add_node_data():
    #创建初始粒子
    nodes = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    node_set = NodeSet(nodes)
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

if __name__ == "__main__":
    test_neighbors()
    test_add_node_data()